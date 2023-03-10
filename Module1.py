import ee
import BackScatter as scatter
import matplotlib.pyplot as plt
import FeatureExtract as fe
from csv import writer
# Trigger the authentication flow.
#ee.Authenticate()

# Initialize the library.
ee.Initialize()

vv_snr = []
vh_snr = []

#2018, 2019, 2020, 2021
year = 2018
ivec_array = []
dataset_csv = 'feature_sets.csv'

for count in range(1, 12) :
    try :
        print('Processing %d' % (count))
        # Import the MODIS land cover collection.
        lc = ee.ImageCollection('MODIS/006/MCD12Q1')
        
        # Import the MODIS land surface temperature collection.
        lst = ee.ImageCollection('COPERNICUS/S1_GRD')
        #lst = ee.ImageCollection('COPERNICUS/S1_GRD')
        
        # Import the USGS ground elevation image.
        elv = ee.Image('USGS/SRTMGL1_003')
        
        i_date = str(year) + "-" + str(count) + "-01"
        f_date = str(year) + "-"+ str(count) + "-28"
        
        # Initial date of interest (inclusive).
        #i_date = i_dates[count]
        
        # Final date of interest (exclusive).
        #f_date = f_dates[count]
        
        # Selection of appropriate bands and dates for LST.
        lst = lst.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW')).select('VV').filterDate(i_date, f_date)
        
        # Define the urban location of interest as a point near given area
        u_lon = 77.5686
        u_lat = 19.9104
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        
        # Define the rural location of interest as a point away from the city.
        r_lon = u_lon
        r_lat = u_lat
        r_poi = ee.Geometry.Point(r_lon, r_lat)
        
        scale = 1000  # scale in meters
        
        # Print the elevation near pusad, Maharashtra.
        elv_urban_point = elv.sample(u_poi, scale).first().get('elevation').getInfo()
        print('Ground elevation at urban point:', elv_urban_point, 'm')
        
        # Calculate and print the mean value of the LST collection at the point.
        lst_urban_point = lst.mean().sample(u_poi, scale).first().get('VV').getInfo()
        print('Average daytime LST at urban point:', round(lst_urban_point*0.02 -273.15, 2), '°C')
        
        # Print the land cover type at the point.
        lc_urban_point = lc.first().sample(u_poi, scale).first().get('LC_Type1').getInfo()
        print('Land cover value at urban point is:', lc_urban_point)
        
        import pandas as pd
        
        def ee_array_to_df(arr, list_of_bands):
            """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
            df = pd.DataFrame(arr)
        
            # Rearrange the header.
            headers = df.iloc[0]
            df = pd.DataFrame(df.values[1:], columns=headers)
        
            # Remove rows without data inside.
            df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()
        
            # Convert the data to numeric values.
            for band in list_of_bands:
                df[band] = pd.to_numeric(df[band], errors='coerce')
        
            # Convert the time field into a datetime.
            df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        
            # Keep the columns of interest.
            df = df[['time','datetime',  *list_of_bands]]
        
            return df
        
        #lst_df_urban = ee_array_to_df(lst_u_poi,['VV'])
        
        def t_modis_to_celsius(t_modis):
            """Converts MODIS LST units to degrees Celsius."""
            t_celsius =  0.02*t_modis - 273.15
            return t_celsius
        
        
        import numpy as np
        
        ## Then, define the fitting function with parameters.
        def fit_func(t, lst0, delta_lst, tau, phi):
            return lst0 + (delta_lst/2)*np.sin(2*np.pi*t/tau + phi)
        
        roi = u_poi.buffer(1e6)
        
        # Reduce the LST collection by mean.
        lst_img = lst.mean()
        
        # Adjust for scale factor.
        lst_img = lst_img.select('VV').multiply(0.02)
        
        # Convert Kelvin to Celsius.
        lst_img = lst_img.select('VV').add(-273.15)
        
        # Create a buffer zone of 10 km around Pusad.
        pusad = u_poi.buffer(10000)  # meters
        
        #snic = ee.Algorithms.Image.Segmentation.KMeans(lst_img, 4);
        
        link = lst_img.getDownloadURL({
            'scale': 20,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF',
            'region': pusad})
        print(link)
        
        print('Downloading...')
        zip_name = "downloaded.zip"
        out_folder = 'extracted'
        out_file = out_folder + "/"+ 'download.VV.tif';
        proc_file = 'output.tif'
        
        import urllib.request
        urllib.request.urlretrieve(link, zip_name)
        
        print('Extracting...')
        import zipfile
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(out_folder)
            
        print('Files unzipped to ' + out_folder + '/ folder')
        
        from skimage import io
        from skimage.transform import resize
        
        # read the image stack
        img = io.imread(out_file)
        
        folderName = ''
        if((count >= 1 and count <= 3) or (count >= 11 and count <= 12) ) :
            folderName = 'dataset/type1/'
        elif(count >= 4 and count <= 7 ) :
            folderName = 'dataset/type2/'
        else :
            folderName = 'dataset/type3/'
        
        fname = folderName + i_date + "_seg.png"
        
        img2 = (img-np.amin(img))/(np.amax(img)-np.amin(img))
        img2 = np.around(img2*255)
        img2 = img2.astype(np.uint8)
        img2 = resize(img2, (150, 150), anti_aliasing=True)
        img3 = np.zeros((150, 150, 3))
        img3[:,:,0] = img2;
        img3[:,:,1] = img2;
        img3[:,:,2] = img2;
        
        io.imsave(fname, img3)
        
        f1 = fe.findFourierWavelet(fname)
        print('Fourier & Wavelet Features Extracted...')
        f1 = np.concatenate( (f1, fe.findDCTConv(fname).ravel() ) )
        print('DCT & Convolutional Features Extracted...')
        f1 = np.concatenate( (f1, fe.findMFCC(fname) ) )
        print('MFCC Features Extracted...')
        
        f1 = np.concatenate( (f1, [fname] ) )
        
        ivec_array.append(f1)
        
        # show the image
        plt.imshow(img)
        plt.axis('off')
        # save the image
        plt.savefig(proc_file, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.0)
        (hh, vh, vv, hv) = scatter.findBackScatterCoefficients(proc_file, True)
        
        snr_vh = scatter.PSNR(hh, vh)
        snr_vv = scatter.PSNR(hh, vv)
    
        vv_snr.append(snr_vv)
        vh_snr.append(snr_vh)
    except :
        print('Continue...')
    
    #totTime = 0
    #for tcount in range(0, totTime) :
    #    time.sleep(1)
    #    print('Sleep %d of %d' % (tcount+1, totTime))
plt.plot(vv_snr)
plt.show()
plt.figure()
plt.plot(vh_snr)
plt.show()

#Write this to the file
with open(dataset_csv, 'a', newline='') as f_object:
    writer_object = writer(f_object)
    for iCount in range(0, len(ivec_array)) :
        writer_object.writerow(ivec_array[iCount])
    
    f_object.close()
    print('CSV Written')