use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use tch::{self, Tensor, data::Iter2};


#[derive(Debug)]
pub struct Dataset {
    pub train_x: Tensor,
    pub train_y: Tensor,
    pub test_x: Tensor,
    pub test_y: Tensor,
    pub validation_x: Tensor,
    pub validation_y: Tensor,
}

impl Dataset {
    pub fn train_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.train_x, &self.train_y, batch_size)
    }

    pub fn test_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.test_x, &self.test_y, batch_size)
    }

    pub fn validation_iter(&self, batch_size: i64) -> Iter2 {
        Iter2::new(&self.validation_x, &self.validation_y, batch_size)
    }

    pub fn from_vec(records: Vec<[f64; 4]>, input_count: usize, output_count: usize, ) -> Dataset {
        
        // get file reader
        let mut x: Vec<Vec<f32>> = vec![];
        let mut y: Vec<Vec<f32>> = vec![];
        
        // 2. read file and split into input/output vectors
        for record in records.iter() {
            let mut ins = vec![];
            let mut outs = vec![];
            for r in record.iter().take(input_count) {
                ins.push(*r as f32);
            }
            for r in record.iter().skip(input_count).take(output_count) {
                outs.push(*r as f32);
            }
            x.push(ins);
            y.push(outs);
        }

        // Do normalization here unless already normalized or smth?

        // 3. split input output into train/test/validation vectors
        const TRAIN_PERCENTAGE: f32 = 0.8; // 80% train data
        const TEST_PERCENTAGE: f32 = 0.1; // 10% test and 10% validation
        let train_split = (TRAIN_PERCENTAGE * x.len() as f32) as usize;
        let test_split = ((TRAIN_PERCENTAGE+TEST_PERCENTAGE) * x.len() as f32) as usize;

        let mut train_x: Vec<&[f32]> = vec![];
        let mut train_y: Vec<&[f32]> = vec![];
        let mut test_x: Vec<&[f32]>  = vec![];
        let mut test_y: Vec<&[f32]> = vec![];
        let mut valid_x: Vec<&[f32]>  = vec![];
        let mut validation_y: Vec<&[f32]> = vec![];

        for i in 0..x.len() {
            if i < train_split {
                train_x.push(x[i].as_slice());
                train_y.push(y[i].as_slice());
            } else if i < test_split {
                test_x.push(x[i].as_slice());
                test_y.push(y[i].as_slice());
            } else {
                valid_x.push(x[i].as_slice());
                validation_y.push(y[i].as_slice());
            }
        }

        // Convert to Tensors and TODO: send to device if CUDA works?
        let train_x = Tensor::of_slice2(train_x.as_slice());
        let test_x = Tensor::of_slice2(test_x.as_slice());
        let validation_x = Tensor::of_slice2(valid_x.as_slice());
        let train_y = Tensor::of_slice2(train_y.as_slice());
        let test_y = Tensor::of_slice2(test_y.as_slice());
        let validation_y = Tensor::of_slice2(validation_y.as_slice());

        Dataset {
            train_x,
            train_y,
            test_x,
            test_y,
            validation_x,
            validation_y,
        }
    }


}



use noise::{NoiseFn, SuperSimplex};
use rand::{self, Rng};

pub fn generate_raw_dataset(samples: usize) -> Vec<[f64; 4]> {
    const NOISE_FN_SCALE : f64 = 2.0;

    let noise_function = SuperSimplex::new(1337);

    let mut rng = rand::thread_rng();

    
    let mut avg_out = 0.0f64;
    let mut min_out = 0.0f64;
    let mut max_out = 0.0f64;
    // we generate a dataset that has 3 inputs and 1 output
    let mut records: Vec<[f64;4]> = vec![];
    for _ in 0..=samples {
        let ins: &[f64;3] = &[rng.gen(), rng.gen(), rng.gen()];
        
        let scaled_ins = ins.map(|i| i * NOISE_FN_SCALE);
        
        let out : f64 = noise_function.get(scaled_ins);
        avg_out+=out;
        if out < min_out {
            min_out = out;
        }
        if out > max_out {
            max_out = out;
        }
        records.push([ins[0], ins[1], ins[2], out]);
        
    }
    avg_out /= samples as f64;
    println!("avg={}, min={}, max={}", avg_out, min_out, max_out);
    let data: Vec<[f64; 4]> = records.par_iter()
        .map(|&[a,b,c,d]| 
            [a,b,c,(d - min_out) / (max_out + min_out.abs())]
        ).collect();
    data

}

