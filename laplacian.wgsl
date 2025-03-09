struct ImageDimensions {
    width: u32,
    height: u32,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> dimensions: ImageDimensions;

// Function to access 2D pixel data from 1D array
fn get_pixel(x: u32, y: u32) -> f32 {
    let index = y * dimensions.width + x;
    return input_image[index];
}

// Main compute shader function
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Ensure we're within image boundaries (excluding edges)
    if (x == 0u || y == 0u || x >= (dimensions.width - 1u) || y >= (dimensions.height - 1u)) {
        return;
    }
    
    // Apply Laplacian filter 
    let center = get_pixel(x, y);
    let top = get_pixel(x, y - 1u);
    let bottom = get_pixel(x, y + 1u);
    let left = get_pixel(x - 1u, y);
    let right = get_pixel(x + 1u, y);
    
    // Laplacian value at this pixel
    let laplacian = -4.0 * center + top + bottom + left + right;
    
    // Store the squared result for variance calculation
    let index = y * dimensions.width + x;
    output_data[index] = laplacian * laplacian;
}