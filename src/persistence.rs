use crate::model::Model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// A checkpoint containing the model state and associated training metadata.
#[derive(Serialize, Deserialize)]
pub struct ModelCheckpoint {
    /// The spiking neural network model
    pub model: Model,
    /// Optional metadata about the training run (e.g., "epoch", "learning_rate", "accuracy")
    pub metadata: HashMap<String, String>,
    /// Version of the checkpoint format
    pub version: String,
}

impl ModelCheckpoint {
    /// Creates a new checkpoint for a model with optional metadata.
    pub fn new(model: Model, metadata: HashMap<String, String>) -> Self {
        Self {
            model,
            metadata,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Saves the checkpoint to a JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads a checkpoint from a JSON file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::LIFNeuron;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_load_checkpoint() {
        let mut model = Model::new(2, 2, LIFNeuron::default());
        model.set_weight(0, 0, 0.5);
        model.set_weight(1, 1, 0.8);

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("accuracy".to_string(), "0.95".to_string());

        let checkpoint = ModelCheckpoint::new(model, metadata);

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save
        checkpoint.save(path).expect("Failed to save checkpoint");

        // Load
        let loaded = ModelCheckpoint::load(path).expect("Failed to load checkpoint");

        // Verify model
        assert_eq!(loaded.model.neurons.len(), 2);
        assert_eq!(loaded.model.weights[0][0], 0.5);
        assert_eq!(loaded.model.weights[1][1], 0.8);

        // Verify metadata
        assert_eq!(loaded.metadata.get("epoch").unwrap(), "10");
        assert_eq!(loaded.metadata.get("accuracy").unwrap(), "0.95");
        assert_eq!(loaded.version, env!("CARGO_PKG_VERSION"));
    }
}
