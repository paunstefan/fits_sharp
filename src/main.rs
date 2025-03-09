use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::error::Error;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::io::Read;

use fitsio::FitsFile;
use ndarray::Array2;
use wgpu::util::DeviceExt;
use futures::executor::block_on;
use rayon::prelude::*;

// Define the image score struct
#[derive(Debug, PartialEq,Clone)]
struct ImageScore {
    path: PathBuf,
    clarity_score: f64,
}

// Custom ordering implementation for sorting
impl Ord for ImageScore {
    fn cmp(&self, other: &Self) -> Ordering {
        other.clarity_score.partial_cmp(&self.clarity_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ImageScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ImageScore {}

// GPU context for wgpu processing
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Get the directory path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <directory_with_fits_files>", args[0]);
        return Err("Invalid number of arguments".into());
    }
    
    let dir_path = &args[1];
    let fits_files = find_fits_files(dir_path)?;
    
    if fits_files.is_empty() {
        println!("No FITS files found in directory: {}", dir_path);
        return Ok(());
    }

    // Initialize GPU context
    let gpu_context = match initialize_gpu() {
        Ok(context) => Arc::new(context),
        Err(e) => {
            eprintln!("Failed to initialize GPU, falling back to CPU: {}", e);
            return fallback_to_cpu(fits_files);
        }
    };
    
    // Process files in parallel using Rayon, but offload computation to GPU
    let gpu_context_ref = Arc::clone(&gpu_context);
    let results = Arc::new(Mutex::new(Vec::with_capacity(fits_files.len())));
    
    fits_files.par_iter().for_each(|file_path| {
        match process_image_on_gpu(file_path, &gpu_context_ref) {
            Ok(score) => {
                if let Ok(mut results_vec) = results.lock() {
                    results_vec.push(ImageScore {
                        path: file_path.clone(),
                        clarity_score: score,
                    });
                }
            },
            Err(e) => {
                eprintln!("Error processing {} on GPU: {}", file_path.display(), e);
            }
        }
    });
    
    // Extract results and sort
    let mut image_scores = results.lock().unwrap().clone();
    image_scores.sort();
    
    // Print sorted files (from least blurry to most blurry)
    for score in image_scores {
        println!("{}", score.path.display());
    }
    
    Ok(())
}

fn find_fits_files(dir_path: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let entries = fs::read_dir(dir_path)?
        .collect::<Result<Vec<_>, _>>()?;
    
    // Process directory entries in parallel
    let fits_files: Vec<PathBuf> = entries.par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            
            if path.is_file() {
                // Check for FITS file extensions (.fits, .fit, .fts)
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "fits" || ext_str == "fit" || ext_str == "fts" {
                        return Some(path.clone());
                    }
                }
            }
            None
        })
        .collect();
    
    Ok(fits_files)
}

// Load shader code from file
fn load_shader(filename: &str) -> Result<String, Box<dyn Error>> {
    let mut shader_file = fs::File::open(filename)?;
    let mut shader_code = String::new();
    shader_file.read_to_string(&mut shader_code)?;
    Ok(shader_code)
}

fn initialize_gpu() -> Result<GpuContext, Box<dyn Error>> {
    // Initialize wgpu
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    // Request adapter
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).ok_or("Failed to find an appropriate adapter")?;
    
    // Create device and queue
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))?;
    
    // Create bind group layout for our compute shader
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            // Input buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output buffer (will store variance)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniform buffer for image dimensions
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("bind_group_layout"),
    });
    
    // Try to load shader from file, with fallback to embedded shader
    let shader_src = match load_shader("laplacian.wgsl") {
        Ok(src) => src,
        Err(e) => {
            eprintln!("Warning: Failed to load shader from file ({}), using embedded fallback", e);
            include_str!("laplacian.wgsl").to_string()
        }
    };
    
    // Create shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Laplacian Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    
    // Create pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    
    Ok(GpuContext {
        device,
        queue,
        compute_pipeline,
        bind_group_layout,
    })
}

fn process_image_on_gpu(file_path: &Path, gpu_context: &GpuContext) -> Result<f64, Box<dyn Error>> {
    // Open the FITS file
    let mut fitsfile = FitsFile::open(file_path)?;
    
    // Read the primary HDU
    let hdu = fitsfile.primary_hdu()?;
    
    // Get image dimensions from FITS header
    let naxis = hdu.read_key::<i64>(&mut fitsfile, "NAXIS")?;
    if naxis < 2 {
        return Err("Image has fewer than 2 dimensions".into());
    }
    
    // Get the dimensions of each axis
    let width = hdu.read_key::<i64>(&mut fitsfile, "NAXIS1")? as usize;
    let height = hdu.read_key::<i64>(&mut fitsfile, "NAXIS2")? as usize;
    
    
    // Read image data
    // Fitsio reads data as a flat Vec, we'll reshape it to a 2D array
    let raw_data: Vec<f32> = hdu.read_image(&mut fitsfile)?;
    
    // Reshape the raw data into a 2D ndarray
    let image_data = Array2::from_shape_vec((height, width), raw_data)?;
    let flat_data: Vec<f32> = image_data.iter().cloned().collect();
    
    // Create input buffer
    let input_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&flat_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let width_u32 = width as u32;
    let height_u32 = height as u32;
    
    // Create output buffer (for Laplacian squared values)
    let output_buffer_size = (width_u32 * height_u32 * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
    let output_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // Create staging buffer (for reading back results)
    let staging_buffer = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    
    // Create uniform buffer for dimensions
    let dimensions_data = [width_u32, height_u32];
    let uniform_buffer = gpu_context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&dimensions_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Create bind group
    let bind_group = gpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &gpu_context.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
        label: Some("compute_bind_group"),
    });
    
    // Create command encoder
    let mut encoder = gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });
    
    // Dispatch compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Laplacian Compute Pass"),
        });
        compute_pass.set_pipeline(&gpu_context.compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        // Calculate workgroup counts (ceiling division)
        let workgroup_count_x = (width_u32 + 15) / 16;
        let workgroup_count_y = (height_u32 + 15) / 16;
        
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }
    
    // Copy the result to staging buffer
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    
    // Submit the work
    gpu_context.queue.submit(std::iter::once(encoder.finish()));
    
    // Read back the result
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    
    // Poll the device to perform the mapping operation
    gpu_context.device.poll(wgpu::Maintain::Wait);
    
    // Await the mapping
    if let Ok(Ok(())) = block_on(receiver) {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        // Calculate variance from the result
        if !result.is_empty() {
            let sum: f32 = result.iter().sum();
            let mean = sum / result.len() as f32;
            
            // Variance is already pre-computed (we stored squared differences)
            Ok(mean as f64)
        } else {
            Err("Empty result from GPU computation".into())
        }
    } else {
        Err("Failed to read back from GPU".into())
    }
}

// Fallback to CPU computation if GPU isn't available or fails
fn fallback_to_cpu(fits_files: Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    println!("Using CPU fallback for processing");
    
    // Process all files in parallel and calculate clarity scores
    let mut image_scores: Vec<_> = fits_files.par_iter()
        .filter_map(|file_path| {
            match calculate_clarity_score_cpu(file_path) {
                Ok(score) => Some(ImageScore {
                    path: file_path.clone(),
                    clarity_score: score,
                }),
                Err(e) => {
                    eprintln!("Error processing {}: {}", file_path.display(), e);
                    None
                }
            }
        })
        .collect();
    
    // Sort the scores (highest clarity score first)
    image_scores.sort();
    
    // Print sorted files (from least blurry to most blurry)
    for score in image_scores {
        println!("{}", score.path.display());
    }
    
    Ok(())
}

fn calculate_clarity_score_cpu(file_path: &Path) -> Result<f64, Box<dyn Error>> {
    // Open the FITS file
    let mut fitsfile = FitsFile::open(file_path)?;
    
    // Read the primary HDU
    let hdu = fitsfile.primary_hdu()?;
    
    // Get image dimensions from FITS header
    let naxis:i64 = hdu.read_key(&mut fitsfile, "NAXIS")?;
    if naxis < 2 {
        return Err("Image has fewer than 2 dimensions".into());
    }
    
    // Get the dimensions of each axis
    let width = hdu.read_key::<i64>(&mut fitsfile, "NAXIS1")? as usize;
    let height = hdu.read_key::<i64>(&mut fitsfile, "NAXIS2")? as usize;
    
    // Read image data
    // Fitsio reads data as a flat Vec, we'll reshape it to a 2D array
    let raw_data: Vec<f32> = hdu.read_image(&mut fitsfile)?;
    
    // Reshape the raw data into a 2D ndarray
    let image_data = Array2::from_shape_vec((height, width), raw_data)?;
    
    // Calculate clarity using Laplacian variance (higher variance = less blurry)
    let (height, width) = image_data.dim();
    
    if height <= 2 || width <= 2 {
        return Err("Image too small for Laplacian calculation".into());
    }
    
    let mut laplacian_values = Vec::with_capacity((height - 2) * (width - 2));
    
    for y in 1..height-1 {
        for x in 1..width-1 {
            let center = image_data[[y, x]];
            let top = image_data[[y-1, x]];
            let bottom = image_data[[y+1, x]];
            let left = image_data[[y, x-1]];
            let right = image_data[[y, x+1]];
            
            // Apply Laplacian kernel
            let lap_value = -4.0 * center + top + bottom + left + right;
            laplacian_values.push(lap_value);
        }
    }
    
    // Calculate mean
    let sum: f32 = laplacian_values.iter().sum();
    let mean = sum / (laplacian_values.len() as f32);
    
    // Calculate variance
    let variance_sum: f32 = laplacian_values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum();
    
    let variance = variance_sum / (laplacian_values.len() as f32);
    
    Ok(variance as f64)
}