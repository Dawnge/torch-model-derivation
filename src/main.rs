
mod datasets;
use datasets::*;

mod models;
use models::*;

use tch::{nn::{self, Optimizer, OptimizerConfig, Module}, Device, Reduction, Tensor, Cuda};
use tensorboard_rs::summary_writer::SummaryWriter;





const MAX_EPOCHS: usize = 500;

fn main() {
    tch::maybe_init_cuda();
    const SAMPLES: usize = 100_000;

    let dataset = Dataset::from_vec(
        generate_raw_dataset(100_000),
        3, 
        1
    );
   
    const INPUTS: i64 = 3;
    const OUTPUTS: i64 = 1;
    const HIDDEN_LAYERS: i64 = 2;
    const LAYER_SIZE: i64 = 200;


    const BATCH_SIZE: i64 = 200;
    //let device = Device::cuda_if_available(); // TODO: returns always false for me :(
    println!("CUDA Available: {:?}", Cuda::is_available());
    let vs = nn::VarStore::new(Device::Cpu);
    let net = custom_net(&vs.root(), INPUTS, OUTPUTS, HIDDEN_LAYERS, LAYER_SIZE);

    // initialize tensorboard
    let file_name = format!("./tensorboard/{},{},{},{}", INPUTS, OUTPUTS, HIDDEN_LAYERS, LAYER_SIZE);
    let mut writer = SummaryWriter::new(&file_name);

    let mut opt: Optimizer = nn::sgd(0.9, 0., 0.0001, false).build(&vs, 1e-4).unwrap();

    let scheduler_steps: i64 = MAX_EPOCHS as i64 * (SAMPLES as f64 * 0.8 / BATCH_SIZE as f64) as i64;
    let mut lr_scheduler = OneCycleLr::new(scheduler_steps, (1e-4,1e-2), (0.85, 0.95), 0.1, 0.001);


    // train model:
    for epoch in 0..MAX_EPOCHS {

        let mut epoch_loss = Tensor::from(0.0f64);
        let mut i = 0;

        for (x,y) in dataset.train_iter(BATCH_SIZE) {
            let prediction = net.forward(&x);
            let loss = prediction.mse_loss(&y, Reduction::Mean);
            opt.zero_grad();
            loss.backward();
            opt.step();
            let (lr, mom) = lr_scheduler.step();
            opt.set_lr(lr);
            opt.set_momentum(mom);
            epoch_loss += loss;
            i+=1;
        }
        
        // handle tensorboard output
        writer.add_scalar("Loss/train", (f64::from(epoch_loss) / i as f64) as f32, epoch);
        writer.add_scalar("Opt/LR", lr_scheduler.last_lr as f32, epoch);
        writer.add_scalar("Opt/Momentum", lr_scheduler.last_momentum as f32, epoch);

        writer.flush();

        println!("FINISHED EPOCH, {}", epoch);
    }

}
