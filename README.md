# Building Edge-Influenced GNN For Graph Classification

This research project contains edge-influenced graph neural network layer, GNN models.

Also training pipeline, inference pipeline, etc are added to reproduce the results.

## Installation

You need to install **pytorch** and **pytorch geometric** and other supportive libraries first!. Use
`requirements.txt` or `environment.yaml` file to install the dependencies!

```bash
# For pip installation
pip install -r requirements.txt
# For conda installation
conda env create --file=environment.yaml
```

## Usage
Adjust hyper-parameters as you see fit in the `run.py`

```python
# Run the script for testing 
python3 run.py
# Run the script for evaluation: x times
python3 evaluate.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

* Contact: vimukthirandika1997@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)