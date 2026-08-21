# fast-ice
Detection of landfast ice (fast-ice) from Sentinel-1 SAR imagery

# setup

### Requirements 

1. git
2. pixi package manager (https://pixi.prefix.dev/latest/)

### Install normalized product from source

```bash
git clone https://github.com/jlo031/normalized_covariance.git
cd normalized_covariance
git checkout xarray_update
cd ..
```

### Install the project with pixi

```bash
pixi install
```

### Usage

See the following notebooks for an end to end implementation example:
1. [fast_ice_from_nrb_files.ipynb](notebooks/fast_ice_from_nrb_files.ipynb)
2. [fast_ice_from_stac_xarray.ipynb](notebooks/fast_ice_from_stac_xarray.ipynb)