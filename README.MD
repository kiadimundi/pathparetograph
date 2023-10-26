# Analysis of paths for decision support

IMPORTANT: The code in the `ext` folder is not mine. All respect to the authors of the forceatlas2 library.

IMPORTANT: At first you need to download the dataset from: https://doi.org/10.5281/zenodo.10008220
Unpack it into the `fronts` folder. The structure should look as follows:
```
fronts/
fronts/artificial/...
fronts/realworld/...
fronts/README.md
```

Build the docker image (using `Dockerfile`):
```
docker build -t paretograph .
```

Use an interactive start:
```
docker run --rm -it -p 8050:8050 paretograph
```
Head to http://localhost:8050

Then you can see the output in the console, while you hit the Analyze Buttons in the UI in the browser

Maybe you need to adjust the port.