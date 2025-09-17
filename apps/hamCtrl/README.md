# Hamamatsu Camera Control
Initial Python MagAOX app for controlling the Hamamatsu Quest 2 C15550-22UP qCMOS Camera. Requires that the Dcam SDK and API Lite for Linux have been downloaded and installed from the Hamamatsu resources site.

## Working Controls
_List of controls via cursesINDI interface and there rough estimated status._

| INDI Prop | Short Name | Description | Value Type | Units | Range |  Working Status | 
| ------------- | ------ | ----------- | ------------ | ------ | ------------ | -------- |
| exptime | Exposure Time | Controls setting and viewing Exposure Time. | float | sec | 0.000033949 - 1800.000015185, step = 0.00000001 |Working | 
| hpos | Horizontal Position | ROI horizontal position (~x center pixel of window).  | float | - | 0-4092, step=4 | Working | 
| vpos | Vertical Position | ROI vertical position (~y center pixel of window).  | float | - | 0-2300, step=4 | Working | 
| hsize | Horizontal Size | The 'width' for the ROI | float | - | 4-4096, step=4 | Working |
| vsize | Vertical Size | The 'height' for the ROI | float | - | 4-2304, step=4 | Working |
| binning | Binning | Sets the binning. A higher range is potential but this feature has been 'picky' sometimes. | float | - | 1.0 = 1x1, 2.0 = 2x2, 4.0 = 4x4 | Partly? |
| gain | Contrast Gain | Set the contrast gain | float | - | - | Not Working |
| temp | Temperature | Current temperature of the sensor. | float | C | -50-100 | Working | 
 
_Range values are assuming Dcam is set to SENSORMODE_AREA. Refer to the HTML docs for the 15550-22UP property information for more details._

## Related Docs
_Below are links to related resources for the Hamamatsu qCMOS Camera._

- [Catalog ORCA-Quest 2 qCMOS Camera](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0166E_C15550-22UP.pdf)
- [Driver Software (DCAM)](https://www.hamamatsu.com/jp/en/product/cameras/software/driver-software.html)
- [ORCA-Quest 2 qCMOS Camera- Instruction Manual](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/en/manual/C15550-22UP_IM_En.pdf)
- [Hamamatsu Listing Overview](https://www.hamamatsu.com/jp/en/product/cameras/qcmos-cameras/C15550-22UP.html)