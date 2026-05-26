use pyo3::prelude::*;
use burn::tensor::Tensor;
use burn::backend::NdArray; 
type Backend = NdArray<f32>;

#[pyfunction]
fn solve_oscillator(steps: usize, dt: f32) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let device = Default::default();
    // Initial state: [position, velocity]
    let mut y = Tensor::<Backend, 1>::from_data([1.0, 0.0], &device);
    
    let mut positions = Vec::with_capacity(steps);
    let mut velocities = Vec::with_capacity(steps);

    for _ in 0..steps {
        // Runge-Kutta4 update for the Simple Harmonic Oscillator: dy/dt = [v, -x]
        // A naive unoptimized implementation that overdoes the `.clone()`
        let k1 = oscillator_system(y.clone());
        let k2 = oscillator_system(y.clone() + k1.clone() * (dt / 2.0));
        let k3 = oscillator_system(y.clone() + k2.clone() * (dt / 2.0));
        let k4 = oscillator_system(y.clone() + k3.clone() * dt);

        y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);

        // Extract data to return to Python
        let data = y.to_data().into_vec().ok().unwrap();  
        positions.push(data[0]);
        velocities.push(data[1]);
    }

    Ok((positions, velocities))
}

fn oscillator_system(y: Tensor<Backend, 1>) -> Tensor<Backend, 1> {
    let pos = y.clone().slice([0..1]);
    let vel = y.clone().slice([1..2]);
    Tensor::cat(vec![vel, pos * -1.0], 0)
}

#[pymodule]
fn burn_rk4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_oscillator, m)?)?;
    Ok(())
}