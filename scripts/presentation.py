from flodym import (
    Dimension,
    DimensionSet,
    FlodymArray,
    InflowDrivenDSM,
    LogNormalLifetime,
    MFASystem
)
from flodym.export import PlotlyArrayPlotter
from flodym.mfa_definition import MFADefinition
import pandas as pd

# Preparation
df: pd.DataFrame

# Removing Redundancy: Dimensions
time = Dimension(
    name = "Time",
    letter="t",
    items=[2020, 2025, 2030]
)
region = Dimension(
    name = "Region",
    letter="r",
    items=["UK", "EU", "China"]
)
dims = DimensionSet([time, region])

array = FlodymArray(dims)

array[...] = 1000

# Data Import and Export
array = FlodymArray.from_df(dims, df)

array.to_df().to_csv()

# Plotting
plt = PlotlyArrayPlotter(
    array,
    intra_line_dim="Time",
    linecolor_dim="Region",
    subplot_dim="Product"
)
fig = plt.plot()
plt.show()

# Dynamic Stock Models
dsm = InflowDrivenDSM(
    dims=dims,
    lifetime_model=LogNormalLifetime,
)
dsm.inflow.values[...] = array
dsm.lifetime_model.set_prms(mean=10, std=3)
dsm.compute()

# MFA System
mfa_definition: MFADefinition
dimension_files: dict[str, str]
parameter_files: dict[str, str]
mfa_example = MFASystem.from_csv(
    mfa_definition,
    dimension_files,
    parameter_files
)