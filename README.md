# MIDI-B Challange DKFZ


### Install dependecies
install the dependency packages using poetry

```
poetry install --no-root
```

### Running Jupyter notebooks

* First install jupyter notebook globally using pip. [pip package](https://pypi.org/project/jupyter/)
* Add Current poetry environment to the jupyter notebook kernels by running following command.
`poetry run python -m ipykernel install --user --name dcm-deid`
* Run Jupyter notebook by running following command in terminal.
`jupyter notebook`
* Open the notebooks in Browser from Jupyter notebook starting page. Select the `dcm-deid` kernel in notebook, in case it by default select some other kernel to run those notebooks.
