use std::fs::File;
use std::io::{Read, Result, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone)]
pub struct Dataset {
    pub images: Vec<Vec<f64>>,
    pub labels: Vec<u8>,
}

#[allow(dead_code)]
impl Dataset {
    pub fn new(images: Vec<Vec<f64>>, labels: Vec<u8>) -> Self {
        Self { images, labels }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn split(&self, ratio: f64) -> (Dataset, Dataset) {
        let split_idx = (self.len() as f64 * ratio) as usize;

        let train_images = self.images[..split_idx].to_vec();
        let train_labels = self.labels[..split_idx].to_vec();

        let test_images = self.images[split_idx..].to_vec();
        let test_labels = self.labels[split_idx..].to_vec();

        (
            Dataset::new(train_images, train_labels),
            Dataset::new(test_images, test_labels),
        )
    }

    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();

        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);

        let mut new_images = Vec::with_capacity(self.len());
        let mut new_labels = Vec::with_capacity(self.len());

        for &idx in &indices {
            new_images.push(self.images[idx].clone());
            new_labels.push(self.labels[idx]);
        }

        self.images = new_images;
        self.labels = new_labels;
    }

    pub fn downsample(&mut self, new_dim: usize) {
        if self.images.is_empty() {
            return;
        }

        let old_dim = (self.images[0].len() as f64).sqrt() as usize;
        if old_dim == new_dim {
            return;
        }

        let factor = old_dim / new_dim;
        let mut new_images = Vec::with_capacity(self.len());

        for img in &self.images {
            let mut new_img = Vec::with_capacity(new_dim * new_dim);
            for y in 0..new_dim {
                for x in 0..new_dim {
                    let mut sum = 0.0;
                    for fy in 0..factor {
                        for fx in 0..factor {
                            sum += img[(y * factor + fy) * old_dim + (x * factor + fx)];
                        }
                    }
                    new_img.push(sum / (factor * factor) as f64);
                }
            }
            new_images.push(new_img);
        }

        self.images = new_images;
    }

    pub fn reduce_bit_depth(&mut self, bits: u32) {
        if bits == 0 || bits >= 8 {
            return;
        }

        let levels = 2u32.pow(bits) as f64 - 1.0;
        for img in &mut self.images {
            for val in img.iter_mut() {
                // Quantize to the nearest level in [0.0, 1.0]
                *val = (*val * levels).round() / levels;
            }
        }
    }
}

pub struct MnistLoader;

#[allow(dead_code)]
impl MnistLoader {
    pub fn load(image_path: impl AsRef<Path>, label_path: impl AsRef<Path>) -> Result<Dataset> {
        let images = Self::read_images(image_path, None)?;
        let labels = Self::read_labels(label_path, None)?;

        if images.len() != labels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Number of images and labels do not match",
            ));
        }

        Ok(Dataset::new(images, labels))
    }

    pub fn load_subset(
        image_path: impl AsRef<Path>,
        label_path: impl AsRef<Path>,
        limit: usize,
    ) -> Result<Dataset> {
        let images = Self::read_images(image_path, Some(limit))?;
        let labels = Self::read_labels(label_path, Some(limit))?;

        if images.len() != labels.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Number of images and labels do not match",
            ));
        }

        Ok(Dataset::new(images, labels))
    }

    pub fn load_balanced_subset(
        image_path: impl AsRef<Path>,
        label_path: impl AsRef<Path>,
        limit: usize,
    ) -> Result<Dataset> {
        let all_labels = Self::read_labels(&label_path, None)?;
        let mut label_indices = vec![Vec::new(); 10];

        for (idx, &label) in all_labels.iter().enumerate() {
            if label < 10 {
                label_indices[label as usize].push(idx);
            }
        }

        let samples_per_label = limit / 10;
        let mut selected_indices = Vec::with_capacity(limit);

        for indices in label_indices.iter() {
            let count = indices.len().min(samples_per_label);
            for i in 0..count {
                selected_indices.push(indices[i]);
            }
        }

        // Re-read images efficiently using selected indices
        let mut img_file = File::open(image_path)?;
        let mut magic = [0u8; 4];
        img_file.read_exact(&mut magic)?;
        // Skip num_images, rows, cols
        img_file.seek(SeekFrom::Current(12))?;

        let rows = 28; // Known for MNIST
        let cols = 28;
        let pixels_per_image = rows * cols;

        let mut images = Vec::with_capacity(selected_indices.len());
        let mut labels = Vec::with_capacity(selected_indices.len());

        for &idx in &selected_indices {
            img_file.seek(SeekFrom::Start(16 + (idx * pixels_per_image) as u64))?;
            let mut buffer = vec![0u8; pixels_per_image];
            img_file.read_exact(&mut buffer)?;
            let image: Vec<f64> = buffer.into_iter().map(|p| p as f64 / 255.0).collect();
            images.push(image);
            labels.push(all_labels[idx]);
        }

        Ok(Dataset::new(images, labels))
    }

    fn read_images(path: impl AsRef<Path>, limit: Option<usize>) -> Result<Vec<Vec<f64>>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if u32::from_be_bytes(magic) != 2051 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number for images",
            ));
        }

        let mut num_images_bytes = [0u8; 4];
        file.read_exact(&mut num_images_bytes)?;
        let mut num_images = u32::from_be_bytes(num_images_bytes) as usize;
        if let Some(l) = limit {
            num_images = num_images.min(l);
        }

        let mut rows_bytes = [0u8; 4];
        file.read_exact(&mut rows_bytes)?;
        let rows = u32::from_be_bytes(rows_bytes) as usize;

        let mut cols_bytes = [0u8; 4];
        file.read_exact(&mut cols_bytes)?;
        let cols = u32::from_be_bytes(cols_bytes) as usize;

        let pixels_per_image = rows * cols;
        let mut images = Vec::with_capacity(num_images);

        for _ in 0..num_images {
            let mut buffer = vec![0u8; pixels_per_image];
            file.read_exact(&mut buffer)?;
            let image: Vec<f64> = buffer.into_iter().map(|p| p as f64 / 255.0).collect();
            images.push(image);
        }

        Ok(images)
    }

    fn read_labels(path: impl AsRef<Path>, limit: Option<usize>) -> Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if u32::from_be_bytes(magic) != 2049 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number for labels",
            ));
        }

        let mut num_labels_bytes = [0u8; 4];
        file.read_exact(&mut num_labels_bytes)?;
        let mut num_labels = u32::from_be_bytes(num_labels_bytes) as usize;
        if let Some(l) = limit {
            num_labels = num_labels.min(l);
        }

        let mut labels = vec![0u8; num_labels];
        file.read_exact(&mut labels)?;

        Ok(labels)
    }
}
