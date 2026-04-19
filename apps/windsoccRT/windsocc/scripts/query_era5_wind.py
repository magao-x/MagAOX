import cdsapi

def fetch_lco_wind_data(output_filename="lco_winds.nc"):
    c = cdsapi.Client()

    # LCO coordinates (draw a box)
    Nlat = -28.76
    Wlon = -70.94
    Slat = -29.26
    Elon = -70.44

    # CDS requires a bounding box: [North, West, South, East]
    # For a single point, we just repeat the coordinates
    area_bbox = [Nlat, Wlon, Slat, Elon]

    print("Requesting ERA5 data. This may take a few minutes to queue on their servers...")

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': [
                'u_component_of_wind',
                'v_component_of_wind',
            ],
            'pressure_level': [
                '50', '70', '100', '150', '200', '250',
                '300', '350', '400', '450',
                '500', '550', '600', '650', '700',
                '775', '750', '825', '850', '875', '900',
            ],
            # --- REPLACE THESE WITH YOUR OBSERVATION DATES/TIMES (in UTC!) ---
            'year': '2023',
            'month': '03',
            'day': '10',
            'time': [
                # '04:00', '05:00', '06:00', '07:00', '08:00'
                '06:00'
            ],
            'area': area_bbox,
        },
        output_filename)

    print(f"Data saved successfully to {output_filename}")

if __name__ == "__main__":
    fetch_lco_wind_data()
