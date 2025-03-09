// laplacian_sobel.wgsl - Combined Lunar Image Analysis Shader
@group(0) @binding(0) var<storage, read> inputImage: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputScores: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dimensions: vec2<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let width = dimensions.x;
    let height = dimensions.y;
    
    // Skip edge pixels
    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return;
    }
    
    // Calculate pixel indices for 3x3 neighborhood
    let center_idx = y * width + x;
    let top_idx = (y - 1) * width + x;
    let bottom_idx = (y + 1) * width + x;
    let left_idx = y * width + (x - 1);
    let right_idx = y * width + (x + 1);
    
    // Additional indices for Sobel
    let topleft_idx = (y - 1) * width + (x - 1);
    let topright_idx = (y - 1) * width + (x + 1);
    let bottomleft_idx = (y + 1) * width + (x - 1);
    let bottomright_idx = (y + 1) * width + (x + 1);
    
    // Fetch pixel values
    let center = inputImage[center_idx];
    let top = inputImage[top_idx];
    let bottom = inputImage[bottom_idx];
    let left = inputImage[left_idx];
    let right = inputImage[right_idx];
    let topleft = inputImage[topleft_idx];
    let topright = inputImage[topright_idx];
    let bottomleft = inputImage[bottomleft_idx];
    let bottomright = inputImage[bottomright_idx];
    
    // 1. Calculate Laplacian (4-way neighbors)
    let laplacian = -4.0 * center + top + bottom + left + right;
    
    // 2. Calculate Sobel gradients
    // Horizontal Sobel (Gx)
    let gx = topright + 2.0 * right + bottomright - 
             topleft - 2.0 * left - bottomleft;
    
    // Vertical Sobel (Gy)
    let gy = bottomleft + 2.0 * bottom + bottomright - 
             topleft - 2.0 * top - topright;
    
    // Calculate gradient magnitude
    let sobel = sqrt(gx * gx + gy * gy);
    
    // Store both values in a vec2
    outputScores[center_idx] = vec2<f32>(laplacian * laplacian, sobel);
}