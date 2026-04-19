import xarray as xr
import numpy as np

# Load the dataset
ds = xr.open_dataset("lco_winds.nc")

# 1. Select the specific time (Update this to your actual timestamp)
time_of_interest = '2023-03-10T06:22:00'
time_of_interest_formatted = time_of_interest.replace(':', '')
time_slice = ds.sel(valid_time=time_of_interest,method='nearest')


# 2. Spatially average the u and v components over the lat/lon grid
# This collapses the 2x2 arrays into a single representative value per pressure level
u_mean = time_slice['u'].mean(dim=['latitude', 'longitude']).values
v_mean = time_slice['v'].mean(dim=['latitude', 'longitude']).values
levels = time_slice['pressure_level'].values

# 3. Calculate magnitude and direction using the averaged vectors
speed = np.sqrt(u_mean**2 + v_mean**2)

# Direction the wind is moving TOWARDS
direction_rad = np.arctan2(v_mean, u_mean)
direction_deg = np.degrees(direction_rad) % 360

# 4. Write to a nicely formatted text file
output_filename = f"lco_wind_profile_{time_of_interest_formatted}.txt"

with open(output_filename, 'w') as f:
    header = "Altitude Profile for LCO (Spatial Average)\n"
    divider = "-" * 45 + "\n"
    columns = f"{'Pressure (hPa)':<15} | {'Speed (m/s)':<12} | {'Direction (deg)':<15}\n"
    
    # Write headers to file
    f.write(header)
    f.write(divider)
    f.write(columns)
    f.write(divider)
    
    # Print to console for immediate viewing
    print(header, end='')
    print(divider, end='')
    print(columns, end='')
    print(divider, end='')
    
    # Write and print row data
    for lvl, spd, dir_deg in zip(levels, speed, direction_deg):
        line = f"{lvl:<15.0f} | {spd:<12.1f} | {dir_deg:<15.1f}\n"
        f.write(line)
        print(line, end='')

print(f"\nSuccessfully saved to {output_filename}")