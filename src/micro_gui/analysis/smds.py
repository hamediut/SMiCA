"""
SMDS (Second-order Moment Density Statistics) calculation module.

This module provides functions for calculating two-point correlation functions
and SMDS values for binary microstructure images in 2D and 3D.
"""
import os
import math
import shutil
import subprocess
from glob import glob
import numpy as np
from numba import jit
from typing import Tuple
from tqdm import tqdm

import pandas as pd
from typing import List, Dict, Optional

# import GooseEYE # for 2-point cluster function


@jit(nopython=True)
def two_point_correlation(im: np.ndarray, dim: int, var: int = 1) -> np.ndarray:
    """
    Compute the two-point correlation (second-order moment) for a 2D binary image.

    This function calculates the probability that two points separated by a distance r
    in a specified direction both belong to the same phase (defined by var).

    Args:
        im: 2D binary image array (values should be 0 and 1)
        dim: Direction for correlation calculation
            - 0: x-direction
            - 1: y-direction
        var: Pixel value of the phase of interest (default: 1)

    Returns:
        2D array containing correlation values for each direction and distance

    Note:
        Uses Numba JIT compilation for performance optimization.
    """
    if dim == 0:  # x-direction
        dim_1 = im.shape[1]  # y-axis
        dim_2 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis

    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2 - r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a + r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a + r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r] / float(lmax)
    return two_point


@jit(nopython=True)
def two_point_correlation3D(im: np.ndarray, dim: int, var: int = 1) -> np.ndarray:
    """
    Compute the two-point correlation (second-order moment) for a 3D binary image.

    This function calculates the probability that two points separated by a distance r
    in a specified direction both belong to the same phase (defined by var).

    Args:
        im: 3D binary image array (values should be 0 and 1)
        dim: Direction for correlation calculation
            - 0: x-direction
            - 1: y-direction
            - 2: z-direction
        var: Pixel value of the phase of interest (default: 1)

    Returns:
        3D array containing correlation values for each direction and distance

    Note:
        Uses Numba JIT compilation for performance optimization.
    """
    if dim == 0:  # x-direction
        dim_1 = im.shape[2]  # y-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[2]  # y-axis
    elif dim == 2:  # z-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[2]  # y-axis
        dim_3 = im.shape[1]  # z-axis

    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3 - r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a + r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a + r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a + r, n2]

                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r] / float(lmax)
    return two_point


def calculate_s2(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate S2 (SMDS) values from 2D image data.

    This function computes the two-point correlation in multiple directions
    and returns the averaged S2 values as a function of distance.

    Args:
        image_data: 2D binary image array (values should be 0 and 1)

    Returns:
        1D array of averaged S2 values

    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 2, size=(100, 100))
        >>> s2 = calculate_s2(img)
        >>> print(s2.shape)  # Will be approximately (50,)
    """
    # Take average of directions; use half of the image size in the smallest dimension
    Nr = min(image_data.shape) // 2

    # S2 in x-direction
    two_pt_dim0 = two_point_correlation(image_data, dim=0, var=1)
    # S2 in y-direction
    two_pt_dim1 = two_point_correlation(image_data, dim=1, var=1)

    

    S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
    S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
    S2_average = ((S2_x + S2_y) / 2)[:Nr]

    return S2_average


def calculate_s2_3d(images: np.ndarray, directional: bool = False) -> np.ndarray:
    """
    Calculate average two-point correlation in 3D.

    This function calculates S2 values for 3D image volumes by computing
    correlations in all three principal directions and averaging them.

    Args:
        images: 3D binary image array (values should be 0 and 1)
            Shape: (depth, height, width)
        directional: If True, return directional correlations separately
            (currently not implemented, reserved for future use)

    Returns:
        1D array of averaged S2 values across all three directions

    Note:
        The correlation function is calculated from r=0 to half of the
        smallest dimension in the image. For instance, if the image shape
        is (512, 256, 256), the S2 size will be 128.

    Example:
        >>> import numpy as np
        >>> img_3d = np.random.randint(0, 2, size=(100, 100, 100))
        >>> s2_3d = calculate_s2_3d(img_3d)
        >>> print(s2_3d.shape)  # Will be (50,)
    """
    Nr = min(images.shape) // 2

    two_point_covariance = {}
    for j, direc in enumerate(["x", "y", "z"]):
        two_point_direc = two_point_correlation3D(images, dim=j, var=1)
        two_point_covariance[direc] = two_point_direc

    direc_covariances = {}
    for direc in ["x", "y", "z"]:
        direc_covariances[direc] = np.mean(
            np.mean(two_point_covariance[direc], axis=0), axis=0
        )[:Nr]

    average = (
        np.array(direc_covariances['x']) +
        np.array(direc_covariances['y']) +
        np.array(direc_covariances['z'])
    ) / 3

    return average

def cal_fn( polytope:np.ndarray, n:int)->np.ndarray:
    """This function calculates scaled autocovariance function from Pn function.
    polytope:polytope function it can be two point correlation function (s2) or 
    higher order functions such as p3, p4, etc
    n: order of polytope e.g., n =2 for two-point correlation (s_2), n= 3 for p3h and p3v"""
    numerator = polytope - polytope[0] ** n
    denominator = polytope[0] - polytope[0] ** n
    fn_r = numerator/ denominator
    return fn_r

def calculate_s2_4d(images: np.ndarray):
    """
    Calculate average two-point correlation for a 4D stacks ( a stacks of 3D volumes, not time series).

    This function computes S2 values for 4D image data by calculating
    correlations in all four principal directions and averaging them.

    Args:
        images: 4D binary image array (values should be 0 and 1)
            Shape: (stack_number, depth, height, width)
    Returns:
    df_s2_grouped: a dataframe containing  average S2 values for all 3D volumes in the 4D stack.
    df_f2_grouped: a dataframe containing  average F2 values for all 3D volumes in the 4D stack.
    
    """
    Nr = min(images.shape[1:])//2
    s2_list = []
    f2_list = []

    for i in range(images.shape[0]):
        
        two_point_covariance = {}
        for j, direc in enumerate(["x", "y", "z"]) :
            two_point_direc = two_point_correlation3D(images[i], j, var = 1)
            two_point_covariance[direc] = two_point_direc
    
        direc_covariances = {}
        for direc in ["x", "y", "z"]:
            direc_covariances[direc] = np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]
        s2_average = (direc_covariances['x'] + direc_covariances['y'] + direc_covariances['z'])/3
        s2_list.append(s2_average)
        f2_list.append(cal_fn(s2_average, n = 2))
    # s2--------------------
    df_s2_list = []
    for k in np.arange(0, len(s2_list)):
        df_s2_list.append(pd.DataFrame(s2_list[k], columns = ['s2']))
    df_s2 = pd.concat(df_s2_list)
    df_s2['r'] = df_s2.index
    # df_s2_grouped = df_s2.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )
    df_s2_grouped = df_s2.groupby(['r']).agg( {'s2': ['mean', 'std', 'size'] } )
    
    # f2--------------------
    df_f2_list = []
    for k in np.arange(0, len(f2_list)):
        df_f2_list.append(pd.DataFrame(f2_list[k], columns = ['f2']))
    df_f2 = pd.concat(df_f2_list)
    df_f2['r'] = df_f2.index
    # df_f2_grouped = df_f2.groupby(['r']).agg( {'f2': [np.mean, np.std, np.size] } )
    df_f2_grouped = df_f2.groupby(['r']).agg( {'f2': ['mean', 'std', 'size'] } )

    return df_s2_grouped, df_f2_grouped

def REV(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, pd.DataFrame]:
    
    """
    This function receives a 3D (XCT) image and calculates average S2 and F2 for the whole image and a number of random subvolumes.
    These average correlation functions can then be analysed to determine the REV for the image.
    Parameters
    ----------
    image: np.ndarray
    This is the 3D image read as numpy array to do REV analysis on.

    img_size_list: List
    list of image sizes to calculate correlation functions. These sizes should be smaller than the whole image.

    n_rand_samples: int
    number of random images used for calculating REV. use 30 or more.

    Returns
    --------
    It returns two dictionary: one for s2 (s2_3d_dict) and one for f2 (f2_3d_dict)
    """
    seed =33
    np.random.seed(seed)
    x_max, y_max, z_max = image.shape[:]

    s2_3d_dict = {}
    f2_3d_dict = {}

    for image_size in img_size_list:

        all_crops = np.zeros((n_rand_samples, image_size, image_size, image_size), dtype = np.uint8)
        for i in range(n_rand_samples):

            x = np.random.randint (0, x_max - image_size)
            y = np.random.randint (0, y_max - image_size)
            z = np.random.randint (0, z_max - image_size)

            crop_image = image[x:x + image_size, y:y + image_size, z:z + image_size]
            all_crops[i] = crop_image
            
        df_s2, df_f2 = calculate_s2_4d(all_crops)
        s2_3d_dict[f'sub_{image_size}'] = df_s2
        f2_3d_dict[f'sub_{image_size}'] = df_f2

        # print(f'{image_size} done !')
    
    print('Calculating S2 and F2 for the whole volume ...')
    # calculate S2 and f2 for the whole volume
    s2_avg_original  = calculate_s2_3d(image)
    f2_avg_original = cal_fn(s2_avg_original, n = 2)

    s2_3d_dict['original'] = s2_avg_original
    f2_3d_dict['original'] = f2_avg_original

        
    return s2_3d_dict, f2_3d_dict


## Funnctions for 2D RES analysis

def calculate_two_point_list(images:np.ndarray) -> List[np.ndarray]:
    """
    This function calculates average two-point correlations (s2 and fn) from n_imgs of images
    and returns a list of all of them.
    Parameters:
    images: np.ndarray of shape (n_imgs, img_size, mg_size)
    """
    #Take average of directions; use half linear size assuming equal dimension sizes
    Nr = min(images.shape[1:])//2

    s2_list = []
    f2_list = []
    
    for i in range(images.shape[0]):
        # 1) convert each image in the batch to microstructure
        # 2) calculate the requested polytope function including scaled version
        # 3) append the results to the empty list above
        
        two_pt_dim0 = two_point_correlation(images[i], dim = 0, var = 1) #S2 in x-direction
        # print(type(two_pt_dim0), two_pt_dim0.shape)
        two_pt_dim1 = two_point_correlation(images[i], dim = 1, var = 1) #S2 in y-direction
        # print(type(two_pt_dim1), two_pt_dim1.shape)
        

        S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
        S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
        S2_average = ((S2_x + S2_y)/2)[:Nr]
        
        s2_list.append(S2_average)
        
        # autoscaled covriance---------------------------------------
        # f_average = (S2_average - S2_average[0]**2)/S2_average[0]/(1 - S2_average[0])
        f_average = cal_fn(S2_average, n= 2)
        f2_list.append(f_average)
        
    return s2_list, f2_list


def list_to_df_two_point(s2_list:List[np.ndarray],
                         f2_list:Optional[List[np.ndarray]] = None
                         )-> pd.DataFrame:
    
    df_list = [pd.DataFrame(s2, columns = ['s2'] ) for s2 in s2_list]
    df = pd.concat(df_list)
    df['r'] = df.index
    df_grouped = df.groupby( ['r'] ).agg( {'s2': ['mean', 'std', 'size'] } )

    if f2_list is not None:
            
        df_f2_list = [pd.DataFrame(f2, columns = ['f2'] ) for f2 in f2_list]
        df_f2 = pd.concat(df_f2_list)
        df_f2['r'] = df_f2.index
        df_fn_grouped = df_f2.groupby( ['r'] ).agg( {'f2': ['mean', 'std', 'size'] } ) 

        return df_grouped, df_fn_grouped
    else:
        return df_grouped
    

def RES(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, pd.DataFrame]:
    
    """
    This function receives a 2D binary image and calculates average S2 and F2 for the whole image and a number of random crops.
    These average correlation functions can then be analysed to determine the RES for the image.
    Parameters
    ----------
    image: np.ndarray
    This is the 2D binary image read as numpy array to do RES analysis on.

    img_size_list: List
    list of image sizes to calculate correlation functions. These sizes should be smaller than the whole image.

    n_rand_samples: int
    number of random images used for calculating REV. use 30 or more.

    Returns
    --------
    It returns two dictionary: one for s2 (s2_dict) and one for f2 (f2_dict)
"""
    seed =33
    np.random.seed(seed)
    x_max, y_max = image.shape[:]
    
    s2_dict = {}
    f2_dict = {}


    for image_size in img_size_list:

        all_crops = np.zeros((n_rand_samples, image_size, image_size), dtype = np.uint8)
        for i in range(n_rand_samples):

            x = np.random.randint (0, x_max - image_size)
            y = np.random.randint (0, y_max - image_size)

            crop_image = image[x:x + image_size, y:y + image_size]
            all_crops[i] = crop_image
            
        s2_list, f2_list = calculate_two_point_list(all_crops)
        
        s2_df, f2_df = list_to_df_two_point(s2_list, f2_list)


        s2_dict[f'sub_{image_size}'] = s2_df
        f2_dict[f'sub_{image_size}'] = f2_df

        print(f'Image of size {image_size} done !')

    print('Calculating S2 and F2 for the whole volume ...')
    # calculate S2 and f2 for the whole volume
    s2_avg_original  = calculate_s2(image)
    f2_avg_original = cal_fn(s2_avg_original, n = 2)

    s2_dict['original'] = s2_avg_original
    f2_dict['original'] = f2_avg_original
        
    return s2_dict, f2_dict


###---------------------polytope functions using cpp code------------------------
class Microstructure:
    # init method
    def __init__(self, dims, ns):
        

        self.dims = dims  # number of dimensions (2 or 3)
        self.ns = ns  # number of samples per dimensions - all dims must have equal number of samples

        if dims == 2:  # 2D sample
            self.structure = np.ones((ns, ns))  # stores two-phase (binary) sampled microstructure
            self.sourceimage = np.ones((ns, ns))  # stores source image for self.structure
        elif dims == 3:  # 3D sample
            self.structure = np.ones((ns, ns, ns))
            self.sourceimage = np.ones((ns, ns, ns))
        else:  # only 2D or 3D are allowed
            raise Exception('Number of dimensions must be 2 or 3.')

    # Miscellanous class vars
    name = 'default_sample'  # sample name

    # dims = int(2)
    # ns = int(251)
    # structure = np.ones((ns, ns))
    # sourceimage = np.ones((ns, ns))

    def description(self):
        if self.dims == 2:
            desc_str = "Sample %s is a %dD microstructure with %d x %d pixels." % (
                self.name, self.dims, self.ns, self.ns)
        else:
            desc_str = "Sample %s is a %dD microstructure with %d x %d x %d pixels." % (
                self.name, self.dims, self.ns, self.ns, self.ns)
        return desc_str

    def volumefraction(self):
        self.ninclusion = 0  # number of inclusion/black pixels, assume black pixels have 0 value
        structure = self.structure
        # count inclusion pixels
        if self.dims == 2:
            for ix in range(self.ns):
                for iy in range(self.ns):
                    if structure[ix, iy] == 1: # pore =1, solid = 0
                        self.ninclusion += 1
        elif self.dims == 3:
            for ix in range(self.ns):
                for iy in range(self.ns):
                    for iz in range(self.ns):
                        if structure[ix, iy, iz] == 1:
                            self.ninclusion += 1
        # final volume fraction
        self.volfracvalue = self.ninclusion / (self.ns ** (self.dims))

    def list_inclusion_indeces(self):
        # set up
        self.volumefraction()
        inclist = np.zeros((self.dims, self.ninclusion), dtype=int)  # initiate array
        structure = self.structure  # get structure
        # get inclusion indeces
        iincl = 0
        if self.dims == 2:
            for ix in range(self.ns):
                for iy in range(self.ns):
                    if structure[ix, iy] == 1: #pore =1, solid=0
                        inclist[0, iincl] = ix
                        inclist[1, iincl] = iy
                        iincl += 1
        elif self.dims == 3:
            for ix in range(self.ns):
                for iy in range(self.ns):
                    for iz in range(self.ns):
                        if structure[ix, iy, iz] == 1:
                            inclist[0, iincl] = ix
                            inclist[1, iincl] = iy
                            inclist[2, iincl] = iz
                            iincl += 1
        # output
        self.inclusion_index_list = inclist

    def write_Mconfig(self, file_path=''):
        # check if inclusion list is there
        try:
            inclist = self.inclusion_index_list
        except ValueError:  # if not call listing method
            self.list_inclusion_indeces()
            inclist = self.inclusion_index_list

        # open files
        mconfig = 'Mconfig'
        extension = '.txt'
        myname = self.name
        filename = file_path + myname + '_' + mconfig + extension
        # filename = os.path.join(file_path, myname + '_' + mconfig + extension)

        file = open(filename, 'w')
        # print dims
        # print('%s' % self.ns, file=file)
        # print number of inclusion pixels
        print('%s' % self.ninclusion, file=file)
        # print inclusion index list
        for iincl in range(self.ninclusion):
            print('%s   %s' % (inclist[0, iincl], inclist[1, iincl]), file=file)
        # close file
        file.close()

    # calculating 2-point Correlation Function (S2)
    def estimate_twopoint_correlation(self, file_path='', cppcode_path='', runtime_path=os.getcwd(), verbose=False):
        # file info
        mconfig = 'Mconfig'
        extension = '.txt'
        myname = self.name
        currdir = os.getcwd()

        # Resolve to absolute paths (with the trailing separator these string-concatenation
        # paths expect) BEFORE chdir'ing anywhere. Without this, a caller-supplied relative
        # path (e.g. 'cpp_poly/512/runtime/output/') is interpreted relative to whatever the
        # current directory happens to be *after* the chdir below, not the directory the
        # caller actually meant - which breaks as soon as runtime_path itself is relative.
        file_path = os.path.abspath(file_path) + os.sep
        cppcode_path = os.path.abspath(cppcode_path) + os.sep
        runtime_path = os.path.abspath(runtime_path) + os.sep

        try:
            if runtime_path != currdir:
                os.chdir(runtime_path)

            file1name = runtime_path + mconfig + extension
            file2name = file_path + myname + '_' + mconfig + extension
            codepath = cppcode_path
            outputpath = file_path

            # check if Mconfig files exist
            if os.path.isfile(file2name):
    #             print('%s_Mconfig.txt file exists in: %s' % (self.name, file_path))
    #             print('Mconfig.txt file replaced in current directory')
    #             print('These are assumed to be the same: S2 estimation will proceed.')
                # Copy self.name_Mconfig.txt into Mconfig.txt
                shutil.copyfile(file2name, file1name)
            else:
    #             print('Writing %s_Mconfig.txt file in: %s' % (self.name, file_path))
    #             print('Writing Mconfig.txt file for sample %s in current directory' % (self.name))
                self.write_Mconfig(file_path=outputpath)
                # Copy self.name_Mconfig.txt into Mconfig.txt
                shutil.copyfile(file2name, file1name)

            # check if compiled C++ code is there
            cpp_executable = cppcode_path + 'L-S2_sample.2D'
            if os.path.isfile(cpp_executable):
                pass
            else:
                raise Exception('Executable L-S2_sample.2D not in: %s' % cppcode_path)

            # run C++ code
            cpp_output = subprocess.run(cpp_executable, capture_output=True)
            if verbose:
                print(cpp_output)
            if cpp_output.returncode != 0:
                raise RuntimeError(
                    f"L-S2_sample.2D exited with code {cpp_output.returncode}. "
                    f"stderr: {cpp_output.stderr!r}"
                )

            # load output from file into class attribute
            outputS2_file = runtime_path + 'TS2.txt'
            self.twopoint_corrfunc = np.loadtxt(outputS2_file)
        finally:
            # always return to the original directory, even if something above raised -
            # otherwise a failed call leaves the process cwd changed and every subsequent
            # call (which resolves its own paths relative to cwd) breaks too.
            os.chdir(currdir)

    # calculating n-Polytope functions
    def estimate_npolytope_functions(self, file_path='', cppcode_path='', runtime_path='', verbose=False):
        # file info
        mconfig = 'Mconfig'
        extension = '.txt'
        myname = self.name
        currdir = os.getcwd()

        # Resolve to absolute paths BEFORE chdir'ing anywhere - see the matching comment in
        # estimate_twopoint_correlation() above for why (relative paths silently break once
        # the process cwd has moved to runtime_path).
        file_path = os.path.abspath(file_path) + os.sep
        cppcode_path = os.path.abspath(cppcode_path) + os.sep
        runtime_path = os.path.abspath(runtime_path) + os.sep

        try:
            if runtime_path != currdir:
                os.chdir(runtime_path)

            file1name = runtime_path + mconfig + extension
            file2name = file_path + myname + '_' + mconfig + extension
            codepath = cppcode_path
            outputpath = file_path

            # check if Mconfig files exist
            if os.path.isfile(file2name):
    #             print('%s_Mconfig.txt file exists in: %s' % (self.name, file_path))
    #             print('Mconfig.txt file replaced in current directory')
    #             print('These are assumed to be the same: Pn estimation will proceed.')
                # Copy self.name_Mconfig.txt into Mconfig.txt
                shutil.copyfile(file2name, file1name)
            else:
    #             print('Writing %s_Mconfig.txt file in: %s' % (self.name, file_path))
    #             print('Writing Mconfig.txt file for sample %s in current directory' % (self.name))
                self.write_Mconfig(file_path=outputpath)
                # Copy self.name_Mconfig.txt into Mconfig.txt
                shutil.copyfile(file2name, file1name)

            # check if compiled C++ code is there
            cpp_executable = cppcode_path + '/Sample_Pn_UU'
            if os.path.isfile(cpp_executable):
                pass
            else:
                raise Exception('Executable Sample_Pn_UU not in: %s' % cppcode_path)

            # run C++ code
            cpp_output = subprocess.run(cpp_executable, capture_output=True)
            if verbose:
                print(cpp_output)
            if cpp_output.returncode != 0:
                # subprocess.run() does NOT raise on a non-zero/crashing return code by default,
                # so without this check a failed run (e.g. missing runtime DLLs for this 32-bit
                # MinGW build) silently falls through to loadtxt() below, which then reads
                # whatever old Sobj*.txt files happen to already be sitting in runtime_path.
                raise RuntimeError(
                    f"Sample_Pn_UU exited with code {cpp_output.returncode} (likely a missing "
                    f"runtime DLL, e.g. libgcc_s_dw2-1.dll / libstdc++-6.dll for this 32-bit "
                    f"MinGW build). stderr: {cpp_output.stderr!r}"
                )

            # load output from files into class attributes
            # S2
            outputS2_file = runtime_path + 'sobjS2.txt'
            self.polytope_S2 = np.loadtxt(outputS2_file)
            # L
            outputL_file = runtime_path + 'sobjL.txt'
            self.polytope_L = np.loadtxt(outputL_file)
            # P3V
            outputP3V_file = runtime_path + 'SobjTriV.txt'
            self.polytope_P3V = np.loadtxt(outputP3V_file)
            # P3H
            outputP3H_file = runtime_path + 'SobjTriH.txt'
            self.polytope_P3H = np.loadtxt(outputP3H_file)
            # P4
            outputP4_file = runtime_path + 'SobjSQF.txt'
            self.polytope_P4 = np.loadtxt(outputP4_file)
            # P6V
            outputP6V_file = runtime_path + 'SobjHesaVer.txt'
            self.polytope_P6V = np.loadtxt(outputP6V_file)
            # P8
            #outputP8_file = runtime_path + 'SobjOctagon.txt'
            #self.polytope_P8 = np.loadtxt(outputP8_file)
        finally:
            # always return to the original directory, even if something above raised - see
            # the matching comment in estimate_twopoint_correlation() above.
            os.chdir(currdir)

        # return to current directory when done
        os.chdir(currdir)

    # scaled autocovariance from S2
    def calculate_scaled_autocovariance(self):

        try:  # check for S2 from polytope sampling
            S2 = self.polytope_S2[:, 1]
            pass
        except ValueError:
            raise Exception("No previous S2 exists: first generate S2.")

        # then calculate f(r):
        phi1 = self.volfracvalue   # black phase volume fraction
        phi2 = 1.0 - phi1          # white phase volume fraction
        Xi_of_r = S2 - phi1**2
        f_of_r = Xi_of_r / (phi1 * phi2)
        self.scal_autocov = np.zeros(self.polytope_S2.shape)
        self.scal_autocov[:, 1] = f_of_r
        self.scal_autocov[:, 0] = self.polytope_S2[:, 0]

    # scaled correlations from polytopes
    def calculate_polytope_fn(self):

        try:
            Pn = self.polytope_P3V[:, 1]
        except ValueError:
            raise Exception("Pn functions not found.")

        # P3V
        Pn = self.polytope_P3V[:, 1]
        phi = Pn[0]
        if Pn[-1] != 0.0:
            phi_n = Pn[-1]
        else:
            phi_n = Pn[-2]
        fn = (Pn - phi_n) / (phi - phi_n)
        self.polyfn_P3V = np.zeros(self.polytope_P3V.shape)
        self.polyfn_P3V[:, 1] = fn
        self.polyfn_P3V[:, 0] = self.polytope_P3V[:, 0]

        # P3H
        Pn = self.polytope_P3H[:, 1]
        phi = Pn[0]
        if Pn[-1] != 0.0:
            phi_n = Pn[-1]
        else:
            phi_n = Pn[-2]
        fn = (Pn - phi_n) / (phi - phi_n)
        self.polyfn_P3H = np.zeros(self.polytope_P3H.shape)
        self.polyfn_P3H[:, 1] = fn
        self.polyfn_P3H[:, 0] = self.polytope_P3H[:, 0]

        # P4
        Pn = self.polytope_P4[:, 1]
        phi = Pn[0]
        if Pn[-1] != 0.0:
            phi_n = Pn[-1]
        else:
            phi_n = Pn[-2]
        fn = (Pn - phi_n) / (phi - phi_n)
        self.polyfn_P4 = np.zeros(self.polytope_P4.shape)
        self.polyfn_P4[:, 1] = fn
        self.polyfn_P4[:, 0] = self.polytope_P4[:, 0]

        # P6V
        Pn = self.polytope_P6V[:, 1]
        phi = Pn[0]
        if Pn[-1] != 0.0:
            phi_n = Pn[-1]
        else:
            phi_n = Pn[-2]
        fn = (Pn - phi_n) / (phi - phi_n)
        self.polyfn_P6V = np.zeros(self.polytope_P6V.shape)
        self.polyfn_P6V[:, 1] = fn
        self.polyfn_P6V[:, 0] = self.polytope_P6V[:, 0]


def twoDCTimage2structure_mod(binary_image, par={'name': 'microstructure_from_image', 'begx': 10, 'begy': 10, 'nsamp': 1001, 'edge_buffer': 20,
                                            'thresholding_method': 'otsu', 'thresholding_weight': 1.0, 'nbins': 256,
                                            'make_figs': False, 'fig_res': 400, 'fig_path': ''}):

    #
    # begin by reading in image and check if ndarray class
    if type(binary_image) is np.ndarray:
        pass
    else:
        raise Exception('The input image must be of the numpy.ndarray class.')

    # check dimensions, argument consistency, etc.
    im_shape = binary_image.shape  # get array shape
    im_dims = len(im_shape)  # number of dimensions
    if im_dims != 2:  # dimensions must be 2
        raise Exception('input image must be 2D.')

    output_microstructure = Microstructure(im_dims, par['nsamp'])

    img_binary =binary_image

    output_microstructure.structure = img_binary
    output_microstructure.name = par['name']


    return output_microstructure


def calculate_polytopes(images, par, outputPn, cpathPn, runtimePn, polytope= 's2'):
    
    """this functions calculates polytopes for each image in batch and returns a dataframe with average polytope values.
     it also removes the **Mconfig.txt files in the runtime/output folder.
     we need to delete these files, otherwise it copies the results of previous implementation of the function.
    Inputs:
    images: real or fake batch of images:numpy array (slice, height, width)
    polytope: each of these polytopes can be calculated:
    s2: two-point correlation
    fn: autoscaled s2
    p3h: horizontal triangle
    p3v: vertical triangle
    p4: square
    p6: 
    L: lineal path
    
    Returns:
    1) dataframe containing polytope values
    2) the scaled version of dataframe"""
    
    if len(images.shape) == 3:
        
        
        poly_list = []
        fn_list = [] #scaled version

        image_number = 1
        for i in tqdm(range(images.shape[0])):
            #convert images in each batch into microstructure
            par['name']= f'batch_{image_number}'
            image_number += 1
            img_micr = twoDCTimage2structure_mod(images[i], par)
            img_micr.volumefraction()
            img_micr.list_inclusion_indeces()
            img_micr.estimate_npolytope_functions(file_path=outputPn, cppcode_path=cpathPn, runtime_path=runtimePn,verbose=False)

            img_micr.calculate_scaled_autocovariance()
            img_micr.calculate_polytope_fn()

            if polytope == 's2':
                poly_list.append(img_micr.polytope_S2)
                fn_list.append(img_micr.scal_autocov)
                
            elif polytope == 'p3h':
                poly_list.append(img_micr.polytope_P3H)
                fn_list.append(img_micr.polyfn_P3H)

            elif polytope == 'p3v':
                poly_list.append(img_micr.polytope_P3V)
                fn_list.append(img_micr.polyfn_P3V)

            elif polytope == 'p4':
                poly_list.append(img_micr.polytope_P4)
                fn_list.append(img_micr.polyfn_P4)

            elif polytope == 'p6':
                poly_list.append(img_micr.polytope_P6V)
                fn_list.append(img_micr.polyfn_P6V)
            elif polytope ==  'L':
                poly_list.append(img_micr.polytope_L)
                fn_list.append(img_micr.scal_autocov)
            else:
                raise Exception('Polytope function name is not correct. use one of the s2, p3h, p3v, p4, p6, or L')
        for filename in glob(outputPn + '/batch*'):
            os.remove(filename)         
        return poly_list, fn_list
    
    
    elif len(images.shape) == 2:
        image_number = 1
        
        par['name']= f'batch_{image_number}'
        image_number += 1
        img_micr = twoDCTimage2structure_mod(images, par)
        img_micr.volumefraction()
        img_micr.list_inclusion_indeces()
        img_micr.estimate_npolytope_functions(file_path=outputPn, cppcode_path=cpathPn, runtime_path=runtimePn,verbose=False)

        img_micr.calculate_scaled_autocovariance()
        img_micr.calculate_polytope_fn()
        
        for filename in glob(outputPn + '/batch*'):
            os.remove(filename) 

        if polytope == 's2':
            return img_micr.polytope_S2, img_micr.scal_autocov

        elif polytope == 'p3h':
            return img_micr.polytope_P3H, img_micr.polyfn_P3H

        elif polytope == 'p3v':
            return img_micr.polytope_P3V, img_micr.polyfn_P3V

        elif polytope == 'p4':
            return img_micr.polytope_P4, img_micr.polyfn_P4

        elif polytope == 'p6':
            return img_micr.polytope_P6V, img_micr.polyfn_P6V
        
        elif polytope ==  'L':
            
            return img_micr.polytope_L, img_micr.scal_autocov
        else:
            raise Exception('Polytope function name is not correct. use one of the s2, p3h, p3v, p4,  p6, or L')
            
    for filename in glob(outputPn + '/batch*'):
            os.remove(filename) 



###---------------------two-point cluster function C2 (periodic, no GooseEYE dependency)------------------------
#
# A from-scratch reimplementation of GooseEYE's clusters()/C2() for a single square 2D binary image, so users
# don't need to install GooseEYE (not on PyPI - conda-forge only, a compiled package - exactly the install
# friction this avoids). Matches GooseEYE's documented definition:
#     C2(dx) = P{cluster(x) = cluster(x+dx) != 0}
# i.e. the probability that two points separated by dx belong to the SAME connected cluster of the
# foreground phase (not just the same phase, which is what S2 measures). Clusters are labelled with
# 4-connectivity and periodic boundaries, matching GooseEYE.clusters()'s defaults (its "nearest"
# cross-shaped kernel, periodic=True).

@jit(nopython=True)
def _label_clusters_periodic(binary_image: np.ndarray) -> np.ndarray:
    """
    Label 4-connected clusters of the foreground phase (==1) in a 2D binary image, with periodic
    boundaries (row 0 is adjacent to the last row, column 0 is adjacent to the last column) - matches
    GooseEYE.clusters(img, periodic=True) with its default nearest-neighbour (cross-shaped) kernel.

    Uses an explicit-stack flood fill rather than recursion, since numba doesn't support recursive
    functions well and Python's recursion limit would be hit anyway on a 512x512 image. Because each
    neighbour lookup already wraps around the image edges, a cluster that straddles the boundary is
    discovered in a single flood fill - no separate "stitching" pass is needed.

    Returns an int32 array the same shape as the input: 0 = background, 1..n = cluster id.
    """
    n_rows, n_cols = binary_image.shape
    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    current_label = 0

    # Worst case, every foreground pixel gets pushed onto the stack exactly once (a pixel is only
    # ever pushed the moment its label goes from 0 to non-zero), so this size is always enough.
    stack_r = np.empty(n_rows * n_cols, dtype=np.int64)
    stack_c = np.empty(n_rows * n_cols, dtype=np.int64)

    for start_r in range(n_rows):
        for start_c in range(n_cols):
            if binary_image[start_r, start_c] == 1 and labels[start_r, start_c] == 0:
                current_label += 1
                stack_r[0] = start_r
                stack_c[0] = start_c
                stack_size = 1
                labels[start_r, start_c] = current_label

                while stack_size > 0:
                    stack_size -= 1
                    r = stack_r[stack_size]
                    c = stack_c[stack_size]

                    r_up = r - 1 if r > 0 else n_rows - 1
                    r_down = r + 1 if r < n_rows - 1 else 0
                    c_left = c - 1 if c > 0 else n_cols - 1
                    c_right = c + 1 if c < n_cols - 1 else 0

                    if binary_image[r_up, c] == 1 and labels[r_up, c] == 0:
                        labels[r_up, c] = current_label
                        stack_r[stack_size] = r_up
                        stack_c[stack_size] = c
                        stack_size += 1
                    if binary_image[r_down, c] == 1 and labels[r_down, c] == 0:
                        labels[r_down, c] = current_label
                        stack_r[stack_size] = r_down
                        stack_c[stack_size] = c
                        stack_size += 1
                    if binary_image[r, c_left] == 1 and labels[r, c_left] == 0:
                        labels[r, c_left] = current_label
                        stack_r[stack_size] = r
                        stack_c[stack_size] = c_left
                        stack_size += 1
                    if binary_image[r, c_right] == 1 and labels[r, c_right] == 0:
                        labels[r, c_right] = current_label
                        stack_r[stack_size] = r
                        stack_c[stack_size] = c_right
                        stack_size += 1

    return labels


@jit(nopython=True)
def _cluster_line_pair_counts(labels_line: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    For one periodic 1D line of cluster labels (a row or column), count for each distance
    r = 0..nt-1 how many pairs (i, i+r) [wrapped periodically] belong to the same non-zero
    cluster. Building block for C2(r) - same style as _line_pair_counts() (used for the
    periodic S2 above), just testing cluster-label equality instead of raw phase equality.
    """
    counts = np.zeros(nt, dtype=np.float64)
    for i in range(maxx):
        li = labels_line[i]
        if li == 0:  # background pixel - can never be "in the same cluster" as anything
            continue
        for r in range(nt):
            j = i + r
            if j >= maxx:
                j -= maxx
            if labels_line[j] == li:
                counts[r] += 1.0
    return counts


@jit(nopython=True)
def compute_c2_cluster(labels: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """Two-point cluster function C2(r) = P(same non-zero cluster at distance r), periodic, averaged over rows+columns."""
    total = np.zeros(nt, dtype=np.float64)
    for row in range(maxx):
        total += _cluster_line_pair_counts(labels[row, :], maxx, nt)
    for col in range(maxx):
        total += _cluster_line_pair_counts(labels[:, col], maxx, nt)
    return total / (2.0 * maxx * maxx)


def calculate_c2(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate the two-point cluster function C2(r) for a 2D binary image, without GooseEYE.

    C2(r) = P(two points a distance r apart belong to the same connected cluster of the
    foreground phase), periodic boundaries, 4-connectivity - matches
    GooseEYE.clusters(img, periodic=True) + GooseEYE.C2((Nr, Nr), C, C), radially averaged
    over rows and columns the same way calculate_s2() averages S2.

    Args:
        image_data: 2D square binary image array (values should be 0 and 1)

    Returns:
        1D array of C2 values, r = 0..Nt-1 where Nt = image_size // 2
    """
    if image_data.ndim != 2:
        raise ValueError('calculate_c2 only supports 2D images.')
    if image_data.shape[0] != image_data.shape[1]:
        raise ValueError('calculate_c2 requires a square image.')

    ns = image_data.shape[0]
    nt = ns // 2

    binary = (image_data != 0).astype(np.uint8)
    labels = _label_clusters_periodic(binary)
    c2_values = compute_c2_cluster(labels, ns, nt)

    return c2_values


@jit(nopython=True)
def _s2_native_periodic_line(line: np.ndarray, n: int, nt: int) -> np.ndarray:
    """
    For one periodic 1D line (a row or column), count for each distance r = 0..nt-1 how many
    pairs (i, i+r) [wrapped periodically, modulo the line's own length n] are both foreground.
    Same style as _cluster_line_pair_counts() above, but testing raw phase equality instead of
    cluster-label equality - i.e. this is S2's building block, using the SAME periodic modulus
    (n = image size) as calculate_c2()/_label_clusters_periodic(), unlike compute_s2_polytope()
    below, which uses modulo image_size+1 to match the legacy C++ code.
    """
    counts = np.zeros(nt, dtype=np.float64)
    for i in range(n):
        if line[i] != 1:
            continue
        for r in range(nt):
            j = i + r
            if j >= n:
                j -= n
            if line[j] == 1:
                counts[r] += 1.0
    return counts


@jit(nopython=True)
def _compute_s2_native_periodic(binary_image: np.ndarray, n: int, nt: int) -> np.ndarray:
    """S2(r), periodic modulo the image's own size (matches calculate_c2()'s convention), averaged over rows+columns."""
    total = np.zeros(nt, dtype=np.float64)
    for row in range(n):
        total += _s2_native_periodic_line(binary_image[row, :], n, nt)
    for col in range(n):
        total += _s2_native_periodic_line(binary_image[:, col], n, nt)
    return total / (2.0 * n * n)


def calculate_s2_periodic(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate the periodic two-point correlation S2(r) for a 2D binary image, using the same
    periodic convention (modulo the image's own size, no padding) as calculate_c2(). This is
    NOT the same curve as calculate_s2() (non-periodic/truncating) or the S2 computed
    internally by calculate_polytopes_python() (periodic modulo image_size+1, to match the
    legacy C++ code) - see the module docstring above the polytope section for why those
    differ. Use this one specifically when you need an S2 curve directly comparable to
    calculate_c2()'s output, e.g. for scale_c2_by_connectedness().

    Args:
        image_data: 2D square binary image array (values should be 0 and 1)

    Returns:
        1D array of S2 values, r = 0..Nt-1 where Nt = image_size // 2
    """
    if image_data.ndim != 2:
        raise ValueError('calculate_s2_periodic only supports 2D images.')
    if image_data.shape[0] != image_data.shape[1]:
        raise ValueError('calculate_s2_periodic requires a square image.')

    ns = image_data.shape[0]
    nt = ns // 2

    binary = (image_data != 0).astype(np.uint8)
    return _compute_s2_native_periodic(binary, ns, nt)


def scale_c2_by_connectedness(c2_values: np.ndarray, s2_values: np.ndarray) -> np.ndarray:
    """
    Conditional connectedness ratio g(r) = C2(r) / S2(r): given that two points a distance r
    apart are already known to be in the same phase, the probability that they are also in
    the same connected cluster.

    Unlike scale_by_initial_value() (C2(r)/C2(0), which only guarantees f(0)=1), this ratio
    is bounded in [0, 1] at EVERY r, not just at r=0 - C2(r) <= S2(r) always holds (being in
    the same cluster is a strictly stronger condition than being in the same phase), so the
    ratio can never exceed 1. It also automatically equals 1 at r=0 (C2(0) = S2(0) = phi), so
    it still fits the same "starts at 1" visual convention as the other scaled f(r) curves.

    This is the more physically direct question for understanding connectivity/percolation:
    a value near 1 at some r means the phase stays well-connected out to that distance; a
    value dropping toward 0 means phase-similarity increasingly does NOT imply connectivity.

    Reference: this ratio-based "connectedness" interpretation follows the same spirit as the
    pair-connectedness function used in the continuum-percolation literature, e.g.:
        S. Torquato, J. D. Beasley, Y. C. Chiew, "Two-point cluster function for continuum
        percolation", J. Chem. Phys. 88, 6540 (1988). https://doi.org/10.1063/1.454440
    Note that paper's own pair-connectedness function P2(r) is instead normalized by phi**2
    (i.e. C2(r) = phi**2 * P2(r)), analogous to how S2(r)/phi**2 gives a radial-distribution-
    style g(r) - that quantity does NOT start at 1 at r=0 (it starts at 1/phi), so it doesn't
    match this module's "all scaled curves start at 1" convention the way C2(r)/S2(r) does.
    Dividing pointwise by S2(r) instead (as done here) is the convention used elsewhere in
    the microstructure-reconstruction literature to express connectedness as a conditional
    probability - see e.g. the discussion of C2/S2 in microstructure-reconstruction papers
    such as https://arxiv.org/abs/2402.15815.

    Args:
        c2_values: raw C2(r) array from calculate_c2().
        s2_values: raw S2(r) array from calculate_s2_periodic(), computed on the SAME image -
            not calculate_s2() (non-periodic) and not the polytope module's internal S2
            (periodic modulo image_size+1), since those use different boundary conventions.

    Returns:
        1D array, g(0) = 1.0, bounded in [0, 1] elsewhere. NaN at any r where S2(r) == 0
        exactly (no same-phase pairs at that distance to condition on - e.g. a fragmented
        structure whose largest chord is shorter than r, so the ratio is genuinely
        undefined there, not just numerically unstable).
    """
    with np.errstate(invalid='ignore', divide='ignore'):  # S2(r) == 0 is a real, expected case (see above) - don't warn for it
        return c2_values / s2_values


###---------------------two-point cluster function C2 in 3D (periodic, no GooseEYE dependency)------------------------
#
# Direct 3D extension of the 2D C2 section above - same concept (connected-component labelling
# + periodic pair counting), just with a third axis and 6-connectivity (the 3D analogue of the
# 2D section's 4-connectivity) instead of 4. _cluster_line_pair_counts() and
# _s2_native_periodic_line() are reused as-is below - they already operate on a generic 1D line
# regardless of which image dimensionality it was sliced from.
#
# Memory/performance note: for a large cubic volume (e.g. 512^3), the flood-fill stack arrays
# and the labels array are each sized to the full voxel count, which is a few GB of temporary
# memory for that size (freed once the function returns) - see calculate_c2_3d()'s docstring.

@jit(nopython=True)
def _label_clusters_periodic_3d(binary_image: np.ndarray) -> np.ndarray:
    """
    Label 6-connected clusters of the foreground phase (==1) in a 3D binary image, with
    periodic boundaries in all three dimensions - the 3D analogue of
    _label_clusters_periodic() above (same explicit-stack flood-fill approach, just with a
    third axis and 6 neighbours - up/down/left/right/front/back - instead of 4).

    Returns an int32 array the same shape as the input: 0 = background, 1..n = cluster id.
    """
    n0, n1, n2 = binary_image.shape
    labels = np.zeros((n0, n1, n2), dtype=np.int32)
    current_label = 0

    # Worst case, every foreground voxel gets pushed onto the stack exactly once - same
    # reasoning as the 2D version, just one dimension higher.
    total_voxels = n0 * n1 * n2
    stack_0 = np.empty(total_voxels, dtype=np.int32)
    stack_1 = np.empty(total_voxels, dtype=np.int32)
    stack_2 = np.empty(total_voxels, dtype=np.int32)

    for start_0 in range(n0):
        for start_1 in range(n1):
            for start_2 in range(n2):
                if binary_image[start_0, start_1, start_2] == 1 and labels[start_0, start_1, start_2] == 0:
                    current_label += 1
                    stack_0[0] = start_0
                    stack_1[0] = start_1
                    stack_2[0] = start_2
                    stack_size = 1
                    labels[start_0, start_1, start_2] = current_label

                    while stack_size > 0:
                        stack_size -= 1
                        a = stack_0[stack_size]
                        b = stack_1[stack_size]
                        c = stack_2[stack_size]

                        a_minus = a - 1 if a > 0 else n0 - 1
                        a_plus = a + 1 if a < n0 - 1 else 0
                        b_minus = b - 1 if b > 0 else n1 - 1
                        b_plus = b + 1 if b < n1 - 1 else 0
                        c_minus = c - 1 if c > 0 else n2 - 1
                        c_plus = c + 1 if c < n2 - 1 else 0

                        if binary_image[a_minus, b, c] == 1 and labels[a_minus, b, c] == 0:
                            labels[a_minus, b, c] = current_label
                            stack_0[stack_size] = a_minus
                            stack_1[stack_size] = b
                            stack_2[stack_size] = c
                            stack_size += 1
                        if binary_image[a_plus, b, c] == 1 and labels[a_plus, b, c] == 0:
                            labels[a_plus, b, c] = current_label
                            stack_0[stack_size] = a_plus
                            stack_1[stack_size] = b
                            stack_2[stack_size] = c
                            stack_size += 1
                        if binary_image[a, b_minus, c] == 1 and labels[a, b_minus, c] == 0:
                            labels[a, b_minus, c] = current_label
                            stack_0[stack_size] = a
                            stack_1[stack_size] = b_minus
                            stack_2[stack_size] = c
                            stack_size += 1
                        if binary_image[a, b_plus, c] == 1 and labels[a, b_plus, c] == 0:
                            labels[a, b_plus, c] = current_label
                            stack_0[stack_size] = a
                            stack_1[stack_size] = b_plus
                            stack_2[stack_size] = c
                            stack_size += 1
                        if binary_image[a, b, c_minus] == 1 and labels[a, b, c_minus] == 0:
                            labels[a, b, c_minus] = current_label
                            stack_0[stack_size] = a
                            stack_1[stack_size] = b
                            stack_2[stack_size] = c_minus
                            stack_size += 1
                        if binary_image[a, b, c_plus] == 1 and labels[a, b, c_plus] == 0:
                            labels[a, b, c_plus] = current_label
                            stack_0[stack_size] = a
                            stack_1[stack_size] = b
                            stack_2[stack_size] = c_plus
                            stack_size += 1

    return labels


@jit(nopython=True)
def compute_c2_cluster_3d(labels: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    Two-point cluster function C2(r) for a 3D cubic volume, periodic, averaged over all
    three axes. For every pair of the other two coordinates, extracts a line along each axis
    in turn and reuses _cluster_line_pair_counts() - the same building block the 2D version
    uses, since it just operates on whatever 1D line it's handed.
    """
    total = np.zeros(nt, dtype=np.float64)
    for i in range(maxx):
        for j in range(maxx):
            total += _cluster_line_pair_counts(labels[i, j, :], maxx, nt)  # line along axis 2
            total += _cluster_line_pair_counts(labels[i, :, j], maxx, nt)  # line along axis 1
            total += _cluster_line_pair_counts(labels[:, i, j], maxx, nt)  # line along axis 0
    return total / (3.0 * maxx * maxx * maxx)


def calculate_c2_3d(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate the two-point cluster function C2(r) for a 3D binary volume, without GooseEYE.
    3D analogue of calculate_c2() above - same definition, 6-connectivity instead of 4,
    periodic in all three dimensions.

    Note on cost: this is noticeably heavier than the 2D version. Both the flood-fill
    labelling and the O(3 * n^3 * Nt) pair-counting loop scale with the full voxel count, so
    a 512^3 volume does roughly 500x the work of a 512x512 image - expect a real wait (tens
    of seconds, not milliseconds) and a few GB of temporary memory for the labelling stacks
    on a volume that size. Should be run in a background thread in the GUI, same as the other
    calculations.

    Args:
        image_data: 3D cubic binary image array (values should be 0 and 1)

    Returns:
        1D array of C2 values, r = 0..Nt-1 where Nt = image_size // 2
    """
    if image_data.ndim != 3:
        raise ValueError('calculate_c2_3d only supports 3D images.')
    if not (image_data.shape[0] == image_data.shape[1] == image_data.shape[2]):
        raise ValueError('calculate_c2_3d requires a cubic image (equal size in all 3 dimensions).')

    ns = image_data.shape[0]
    nt = ns // 2

    binary = (image_data != 0).astype(np.uint8)
    labels = _label_clusters_periodic_3d(binary)
    c2_values = compute_c2_cluster_3d(labels, ns, nt)

    return c2_values


@jit(nopython=True)
def _compute_s2_native_periodic_3d(binary_image: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """S2(r) in 3D, periodic modulo the volume's own size (matches calculate_c2_3d()'s convention), averaged over all three axes."""
    total = np.zeros(nt, dtype=np.float64)
    for i in range(maxx):
        for j in range(maxx):
            total += _s2_native_periodic_line(binary_image[i, j, :], maxx, nt)
            total += _s2_native_periodic_line(binary_image[i, :, j], maxx, nt)
            total += _s2_native_periodic_line(binary_image[:, i, j], maxx, nt)
    return total / (3.0 * maxx * maxx * maxx)


def calculate_s2_periodic_3d(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate the periodic two-point correlation S2(r) for a 3D binary volume, using the same
    periodic convention as calculate_c2_3d() (modulo the volume's own size, no padding). 3D
    analogue of calculate_s2_periodic() above - use this (not calculate_s2_3d(), the existing
    non-periodic 3D S2), as the denominator for scale_c2_by_connectedness() on 3D data.

    Args:
        image_data: 3D cubic binary image array (values should be 0 and 1)

    Returns:
        1D array of S2 values, r = 0..Nt-1 where Nt = image_size // 2
    """
    if image_data.ndim != 3:
        raise ValueError('calculate_s2_periodic_3d only supports 3D images.')
    if not (image_data.shape[0] == image_data.shape[1] == image_data.shape[2]):
        raise ValueError('calculate_s2_periodic_3d requires a cubic image (equal size in all 3 dimensions).')

    ns = image_data.shape[0]
    nt = ns // 2

    binary = (image_data != 0).astype(np.uint8)
    return _compute_s2_native_periodic_3d(binary, ns, nt)


###---------------------lineal-path function L in 3D (periodic, native convention)------------------------
#
# The lineal-path function L(r) = P(an entire straight line segment of length r lies fully
# within the foreground phase) is inherently a 1D measurement - it scans straight lines and
# measures chord lengths - so it generalises to 3D exactly like C2 did: scan lines along a
# third axis instead of two, and reuse the same per-line building block unchanged.
# _chord_counts_1d() (defined in the 2D polytope section above, already validated there
# against the real C++ output) needs no changes at all - it just treats whatever 1D array
# it's given as one periodic line, regardless of which 3D axis it was sliced from.
#
# Like calculate_c2_3d()/calculate_s2_periodic_3d() above (and unlike the 2D lineal-path
# inside calculate_polytopes_python(), which uses the polytope module's mod image_size+1
# convention to stay bit-compatible with the legacy C++ Sample_Pn_UU.cpp), this uses the
# native periodic convention (modulo the volume's own size, no padding) - there is no 3D C++
# code to match here, and the native convention gives L(0) == porosity exactly.

@jit(nopython=True)
def compute_L_native_periodic_3d(binary_image: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """Lineal-path function L(r) in 3D, periodic modulo the volume's own size, averaged over all three axes."""
    total = np.zeros(nt, dtype=np.float64)
    for i in range(maxx):
        for j in range(maxx):
            total += _chord_counts_1d_native(binary_image[i, j, :], maxx, nt)  # line along axis 2
            total += _chord_counts_1d_native(binary_image[i, :, j], maxx, nt)  # line along axis 1
            total += _chord_counts_1d_native(binary_image[:, i, j], maxx, nt)  # line along axis 0
    return total / (3.0 * maxx * maxx * maxx)


def calculate_L_3d(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate the lineal-path function L(r) for a 3D binary volume - the probability that an
    entire straight line segment of length r lies fully within the foreground phase. 3D
    analogue of the 2D lineal-path calculation inside calculate_polytopes_python(), scanned
    along three axes (x, y, z) instead of two (rows, columns) - see the module note above for
    why this uses a different (native) periodic convention than the 2D version.

    Unlike C2, L(r) decays to 0 as r grows for any structure that isn't 100% foreground (an
    entire line segment fully inside the foreground phase becomes vanishingly likely once r
    exceeds the largest chord in the material) - it does NOT plateau at a nonzero value the
    way C2 can for a percolating structure, so scale_by_initial_value() (L(r)/L(0)) is the
    appropriate scaling here, not scale_c2_by_connectedness() or scale_polytope_fn().

    Args:
        image_data: 3D cubic binary image array (values should be 0 and 1)

    Returns:
        1D array of L values, r = 0..Nt-1 where Nt = image_size // 2
    """
    if image_data.ndim != 3:
        raise ValueError('calculate_L_3d only supports 3D images.')
    if not (image_data.shape[0] == image_data.shape[1] == image_data.shape[2]):
        raise ValueError('calculate_L_3d requires a cubic image (equal size in all 3 dimensions).')

    ns = image_data.shape[0]
    nt = ns // 2

    binary = (image_data != 0).astype(np.uint8)
    return compute_L_native_periodic_3d(binary, ns, nt)


###---------------------polytope functions, pure Python/numba (no C++ executable)------------------------
#
# This section is a direct, function-by-function port of cpp_poly/512/Cpp_source/Polytope/Sample_Pn_UU.cpp
# for a SINGLE square 2D binary image. It reproduces S2, L (lineal path), P3H/P3V (triangle),
# P4 (square) and P6 (hexagon) without writing Mconfig.txt, without a subprocess call, and without
# depending on an image-size-specific compiled executable (MAXX/Nt are derived from the image at
# runtime instead of being baked in with #define).
#
# One deliberate deviation from the C++: the original wraparound checks compare against MAXX
# ("if (x > MAXX) x -= MAXX;") instead of MAXX-1 ("if (x >= MAXX) ..."). Because C stores the config
# array row-major and contiguous, hitting x == MAXX exactly does not crash - it silently reads the
# next row's pixel 0 instead of wrapping back to this row's pixel 0. That is undefined behaviour, not
# a deliberate periodic-boundary design choice (the real periodic wrap is the MAXX/2 distance folding
# used for S2/L), so here we use the mathematically correct `if x >= maxx: x -= maxx`.
#
# Validated two ways against compare_polytopes.py (repo root):
#  1) against the real sobj*.txt reference files that were already sitting in cpp_poly/512/runtime/,
#     by reconstructing the image from the Mconfig.txt in that same folder;
#  2) against a fresh run of the (statically-relinked) Sample_Pn_UU.exe on a real 512x512 XCT
#     sandstone slice (test_images/XCT_11.4um_binary0000.tif).
# S2, L, P4, P3H, P3V and P6 all matched closely in both runs (max abs diff ~1e-4, zero points differing
# by more than 1e-4) - the tiny residual differences are consistent with the rare pixels where the C++'s
# buggy `>` wraparound and this module's `>=` wraparound disagree, as expected, not a real algorithmic
# mismatch. (Note: the original SobjHesaVer.txt (P6) sitting in cpp_poly/512/runtime/ was itself stale/
# inconsistent with the Mconfig.txt next to it - its R=0 value implied more foreground pixels than that
# image actually has - but the fresh exe run resolved that ambiguity and confirmed compute_p6_polytope
# is correct.)

def _build_padded_config(image: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Build the zero-padded 'config' array the C++ code works on.

    The C++ uses MAXX = image_size + 1 and Nt = image_size // 2 (see the README's compile
    instructions, e.g. MAXX=513/Nt=256 for a 512x512 image). The extra row/column (index
    image_size) is always background - it exists only so the periodic distance-folding
    (`if d >= maxx//2: d = maxx - d`) has an odd-sized period and pixel 0 is never adjacent
    to pixel (image_size - 1) by construction.

    Returns (config, maxx, nt).
    """
    if image.ndim != 2:
        raise ValueError('Input image must be 2D.')
    if image.shape[0] != image.shape[1]:
        raise ValueError('Input image must be square (the C++ code assumes equal dimensions).')

    ns = image.shape[0]
    maxx = ns + 1
    nt = ns // 2

    config = np.zeros((maxx, maxx), dtype=np.uint8)
    config[:ns, :ns] = (image != 0).astype(np.uint8)  # pore/foreground = 1, matches read_config()
    return config, maxx, nt


@jit(nopython=True)
def _line_pair_counts(line: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    For one 1D periodic line (a row or column of `config`), count how many pairs of
    foreground pixels are separated by each distance r = 0..nt-1 (self-pairs count as
    distance 0). Distances wrap around using period `maxx` (matches sampleS2line()/
    sampleS2colume() in the C++). This is the building block for S2(r).
    """
    counts = np.zeros(nt, dtype=np.float64)

    positions = np.empty(maxx, dtype=np.int64)
    n_pos = 0
    for i in range(maxx):
        if line[i] == 1:
            positions[n_pos] = i
            n_pos += 1

    half = maxx // 2
    for i in range(n_pos):
        for j in range(i + 1):  # j <= i: includes the self-pair (distance 0) once per pixel
            d = positions[i] - positions[j]
            if d < 0:
                d = -d
            if d >= half:
                d = maxx - d
            if d < nt:
                counts[d] += 1.0
    return counts


@jit(nopython=True)
def compute_s2_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """Two-point correlation S2(r), periodic-boundary version (matches Sample_Pn_UU.cpp main())."""
    total = np.zeros(nt, dtype=np.float64)
    for col in range(maxx):
        total += _line_pair_counts(config[:, col], maxx, nt)
    for row in range(maxx):
        total += _line_pair_counts(config[row, :], maxx, nt)
    return total / (2.0 * maxx * maxx)


@jit(nopython=True)
def _add_chord_length(counts: np.ndarray, length: int, nt: int) -> None:
    """
    Add one foreground chord of the given length to the sub-chord-length histogram `counts`:
    a chord of length L contains (L - r + 1) sub-intervals of length r, so bin r gets (L - r)
    added for r = 0..L (matches the `N2H[...][r] += (len - r)` loops in the C++).
    """
    for r in range(length + 1):
        if r < nt:
            counts[r] += (length - r)


@jit(nopython=True)
def _chord_counts_1d(line: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    Lineal-path sampling along one periodic 1D line (a row or column of `config`).

    Step 1: classify every pixel by how many of its (periodic) neighbours are foreground:
    -1 = background, 0 = isolated foreground pixel, 1 = chord endpoint, 2 = inside a chord.
    Step 2: collect the endpoints (ener == 1) in array order, then walk them in consecutive
    pairs to get each chord's length, adding it to the histogram via _add_chord_length().
    A chord that straddles the array boundary (pixel 0 and pixel maxx-1 both foreground) is
    stitched together from the last and first endpoints instead of treated as two chords.

    Ported line-for-line from sample_horizontal()/sample_vertical() in Sample_Pn_UU.cpp -
    both functions are the same algorithm, just applied to a row vs. a column.
    """
    counts = np.zeros(nt, dtype=np.float64)

    ener = np.empty(maxx, dtype=np.int64)
    flag_empty = 0
    for i in range(maxx):
        if line[i] == 0:
            ener[i] = -1
        else:
            en = 0
            neb1 = i - 1
            if neb1 < 0:
                neb1 += maxx
            if line[neb1] == 1:
                en += 1
            neb2 = i + 1
            if neb2 >= maxx:
                neb2 -= maxx
            if line[neb2] == 1:
                en += 1
            ener[i] = en
            flag_empty += 1

    position = np.empty(maxx, dtype=np.int64)
    ctp = 0
    for i in range(maxx):
        if ener[i] == 1:
            position[ctp] = i
            ctp += 1
        elif ener[i] == 0:
            counts[0] += 1.0  # isolated single-pixel chord: only contributes to r=0

    if line[0] == 1 and line[maxx - 1] == 1:
        # the foreground run straddles the periodic boundary
        if ctp > 2:
            i = 1
            while i < ctp - 1:
                length = position[i + 1] - position[i] + 1
                _add_chord_length(counts, length, nt)
                i += 2
            # wrap-around chord: from the last endpoint to the end, plus start to the first endpoint
            length = (position[0] + 1) + (maxx - position[ctp - 1])
            _add_chord_length(counts, length, nt)
        elif ctp == 2:
            length = (position[0] + 1) + (maxx - position[ctp - 1])
            _add_chord_length(counts, length, nt)
        elif ctp == 0 and flag_empty != 0:
            # whole line is foreground: one chord that is the full periodic ring
            _add_chord_length(counts, maxx, nt)
    else:
        i = 0
        while i < ctp:
            length = position[i + 1] - position[i] + 1
            _add_chord_length(counts, length, nt)
            i += 2

    return counts


@jit(nopython=True)
def _chord_counts_1d_native(line: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    Same as _chord_counts_1d() above, with one bug fixed: when the ENTIRE periodic line is
    foreground (no background pixel anywhere on it), _chord_counts_1d() adds `maxx - r` to
    each bin `r` - a formula for a *finite* chord of length `maxx` sitting inside a line that
    has a background boundary somewhere. But a fully-foreground periodic line has no boundary
    at all: every one of the `maxx` starting positions admits a valid segment of ANY length r
    (it just keeps going around the ring), so the correct count is a constant `maxx` at every
    r, not a value that decreases with r.

    This formula (`len - r`) is copied verbatim from the original C++ (Sample_Pn_UU.cpp), so
    it isn't a bug introduced by the port - it was already there. It practically never gets
    exercised on real porous-rock images (an entire row/column is essentially never 100%
    foreground), which is why it was never caught during 2D validation against the compiled
    C++ output. _chord_counts_1d() is left untouched (still used by the 2D polytope module,
    compute_L_polytope(), which is deliberately kept bug-compatible with the legacy C++ for
    validated bit-for-bit parity) - this corrected copy is only used by the native (3D, no
    legacy C++ to match) lineal-path code below, where there's no reason to keep the bug.
    """
    counts = np.zeros(nt, dtype=np.float64)

    ener = np.empty(maxx, dtype=np.int64)
    flag_empty = 0
    for i in range(maxx):
        if line[i] == 0:
            ener[i] = -1
        else:
            en = 0
            neb1 = i - 1
            if neb1 < 0:
                neb1 += maxx
            if line[neb1] == 1:
                en += 1
            neb2 = i + 1
            if neb2 >= maxx:
                neb2 -= maxx
            if line[neb2] == 1:
                en += 1
            ener[i] = en
            flag_empty += 1

    position = np.empty(maxx, dtype=np.int64)
    ctp = 0
    for i in range(maxx):
        if ener[i] == 1:
            position[ctp] = i
            ctp += 1
        elif ener[i] == 0:
            counts[0] += 1.0  # isolated single-pixel chord: only contributes to r=0

    if line[0] == 1 and line[maxx - 1] == 1:
        # the foreground run straddles the periodic boundary
        if ctp > 2:
            i = 1
            while i < ctp - 1:
                length = position[i + 1] - position[i] + 1
                _add_chord_length(counts, length, nt)
                i += 2
            # wrap-around chord: from the last endpoint to the end, plus start to the first endpoint
            length = (position[0] + 1) + (maxx - position[ctp - 1])
            _add_chord_length(counts, length, nt)
        elif ctp == 2:
            length = (position[0] + 1) + (maxx - position[ctp - 1])
            _add_chord_length(counts, length, nt)
        elif ctp == 0 and flag_empty != 0:
            # whole line is foreground and periodic - no boundary anywhere, so every one of
            # the `maxx` starting positions admits a chord of any length r (FIX: constant
            # `maxx`, not `maxx - r` - see this function's docstring).
            for r in range(nt):
                counts[r] += maxx
    else:
        i = 0
        while i < ctp:
            length = position[i + 1] - position[i] + 1
            _add_chord_length(counts, length, nt)
            i += 2

    return counts


@jit(nopython=True)
def compute_L_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """Lineal-path function L(r), periodic-boundary version (matches Sample_Pn_UU.cpp main())."""
    total = np.zeros(nt, dtype=np.float64)
    for row in range(maxx):
        total += _chord_counts_1d(config[row, :], maxx, nt)
    for col in range(maxx):
        total += _chord_counts_1d(config[:, col], maxx, nt)
    return total / (2.0 * maxx * maxx)


@jit(nopython=True)
def compute_p4_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    P4 (axis-aligned square) 4-point polytope function.

    For every foreground pixel (r,c) and side length R = 0..nt-1, tests whether the other
    3 corners of the R x R square - (r, c+R), (r+R, c), (r+R, c+R) - are also foreground.
    Ported from squrefunction() in Sample_Pn_UU.cpp (periodic wrap fixed, see module docstring above).
    """
    sq = np.zeros(nt, dtype=np.float64)
    for R in range(nt):
        for r in range(maxx):
            for c in range(maxx):
                if config[r, c] == 1:
                    xx = c + R
                    if xx >= maxx:
                        xx -= maxx
                    yy = r + R
                    if yy >= maxx:
                        yy -= maxx
                    if config[r, xx] == 1 and config[yy, c] == 1 and config[yy, xx] == 1:
                        sq[R] += 1.0
    return sq / (maxx * maxx)


@jit(nopython=True)
def compute_p3v_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    P3-vertical (triangle) 3-point polytope function.

    For every foreground pixel (r,c) and even side length R = 0,2,4,...,nt-1 (odd R are left
    at 0, matching the C++ which only samples even R), tests a triangle with one vertical side
    of length R below (r,c) and an apex offset diagonally by R/2 * sqrt(3): the other two
    vertices are (r+R, c) and (r + R/2, c + R/2*sqrt(3) [+1 rounding term, except at R=0]).
    Ported from TriangleVer() in Sample_Pn_UU.cpp (periodic wrap fixed, see module docstring above).
    """
    tri = np.zeros(nt, dtype=np.float64)
    sqrt3 = math.sqrt(3.0)
    for R in range(0, nt, 2):
        half = R // 2
        for r in range(maxx):
            for c in range(maxx):
                if config[r, c] == 1:
                    if R == 0:
                        xx = c + int(half * sqrt3)
                    else:
                        xx = c + int(half * sqrt3 + 1.0)
                    if xx >= maxx:
                        xx -= maxx
                    gg = r + half
                    if gg >= maxx:
                        gg -= maxx
                    yy = r + R
                    if yy >= maxx:
                        yy -= maxx
                    if config[yy, c] == 1 and config[gg, xx] == 1:
                        tri[R] += 1.0
    return tri / (maxx * maxx)


@jit(nopython=True)
def compute_p3h_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    P3-horizontal (triangle) 3-point polytope function - the row/column mirror of
    compute_p3v_polytope(). Ported from TriangleHor() in Sample_Pn_UU.cpp.
    """
    tri = np.zeros(nt, dtype=np.float64)
    sqrt3 = math.sqrt(3.0)
    for R in range(0, nt, 2):
        half = R // 2
        for r in range(maxx):
            for c in range(maxx):
                if config[r, c] == 1:
                    if R == 0:
                        yy = r + int(half * sqrt3)
                    else:
                        yy = r + int(half * sqrt3 + 1.0)
                    if yy >= maxx:
                        yy -= maxx
                    gg = c + half
                    if gg >= maxx:
                        gg -= maxx
                    xx = c + R
                    if xx >= maxx:
                        xx -= maxx
                    if config[r, xx] == 1 and config[yy, gg] == 1:
                        tri[R] += 1.0
    return tri / (maxx * maxx)


@jit(nopython=True)
def compute_p6_polytope(config: np.ndarray, maxx: int, nt: int) -> np.ndarray:
    """
    P6-vertical (hexagon) 6-point polytope function.

    For every foreground pixel (r,c) and radius R = 0..nt-1, builds the other 5 vertices
    (P2..P6) of a hexagon around (r,c), rounding each coordinate with ceil() exactly as the
    C++ does, and tests whether all 5 are also foreground.
    Ported from HexagonVer() in Sample_Pn_UU.cpp (periodic wrap fixed, see module docstring above).
    Note: HexagonHor()/Octa()/Arbitrary() from the C++ are not ported - calculate_polytopes()
    never used them (only polytope_P6V / HexagonVer feeds the 'p6' result).
    """
    hexv = np.zeros(nt, dtype=np.float64)
    sqrt3 = math.sqrt(3.0)
    for R in range(nt):
        for r in range(maxx):
            for c in range(maxx):
                if config[r, c] == 1:
                    p2x = math.ceil(r + sqrt3 * 0.5 * R)
                    p2y = math.ceil(c - 0.5 * R)
                    p3x = math.ceil(r + sqrt3 * 0.5 * R + sqrt3 * 0.5 * R)
                    p3y = math.ceil(c + 0.5 * R - 0.5 * R)
                    p4x = p3x
                    p4y = math.ceil(c + 0.5 * R - 0.5 * R + R)
                    p5x = p2x
                    p5y = math.ceil(c + R + 0.5 * R)
                    p6x = math.ceil(float(r))
                    p6y = math.ceil(c + R)

                    if p2x >= maxx:
                        p2x -= maxx
                    if p2y < 0:
                        p2y += maxx
                    if p3x >= maxx:
                        p3x -= maxx
                    if p4x >= maxx:
                        p4x -= maxx
                    if p4y >= maxx:
                        p4y -= maxx
                    if p5x >= maxx:
                        p5x -= maxx
                    if p5y >= maxx:
                        p5y -= maxx
                    if p6y >= maxx:
                        p6y -= maxx

                    i2x, i2y = int(p2x), int(p2y)
                    i3x, i3y = int(p3x), int(p3y)
                    i4x, i4y = int(p4x), int(p4y)
                    i5x, i5y = int(p5x), int(p5y)
                    i6x, i6y = int(p6x), int(p6y)

                    if (config[i2x, i2y] == 1 and config[i3x, i3y] == 1 and
                            config[i4x, i4y] == 1 and config[i5x, i5y] == 1 and
                            config[i6x, i6y] == 1):
                        hexv[R] += 1.0
    return hexv / (maxx * maxx)


def scale_polytope_fn(poly_values: np.ndarray) -> np.ndarray:
    """
    Scaled version f(r) of a raw polytope curve Pn(r), in roughly [0, 1]:
        f(r) = (Pn(r) - Pn_inf) / (Pn(0) - Pn_inf)
    where Pn_inf is the long-range plateau value (the last sampled point, or the
    second-to-last if the last is exactly 0.0 - e.g. because the largest distance had no
    valid samples). Matches Microstructure.calculate_polytope_fn() for P3H/P3V/P4/P6V.
    """
    phi = poly_values[0]
    phi_n = poly_values[-1] if poly_values[-1] != 0.0 else poly_values[-2]
    return (poly_values - phi_n) / (phi - phi_n)


def scale_by_initial_value(values: np.ndarray) -> np.ndarray:
    """
    Normalize a curve by its own r=0 value: f(r) = values(r) / values(0).

    Unlike scale_polytope_fn() (which subtracts an assumed long-range plateau - appropriate
    for functions like S2/P3H/P3V/P4/P6 that provably converge to some power of the volume
    fraction at large r), this makes no assumption about what the curve converges to. It
    just rescales so f(0) = 1 and lets the curve decay to wherever it naturally goes.

    Use this for the two-point cluster function C2, whose long-range behaviour is governed
    by percolation (whether some single cluster actually spans the given distance) rather
    than by volume fraction alone - there is no theoretical plateau value to subtract the
    way there is for S2 and the other polytope functions, and subtracting an estimated one
    (scale_polytope_fn's approach) just introduces noise from a single sampled point.
    """
    return values / values[0]


def scale_s2(s2_values: np.ndarray, vol_fraction: float) -> np.ndarray:
    """
    Scaled autocovariance f(r) from S2(r):  f(r) = (S2(r) - phi^2) / (phi * (1 - phi))
    Matches Microstructure.calculate_scaled_autocovariance().
    """
    phi1 = vol_fraction
    phi2 = 1.0 - phi1
    xi = s2_values - phi1 ** 2
    return xi / (phi1 * phi2)


def calculate_polytopes_python(image: np.ndarray,
                                polytopes: Tuple[str, ...] = ('p3h', 'p3v', 'p4', 'p6', 'L')
                                ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Pure-Python/numba replacement for calculate_polytopes() + the compiled Sample_Pn_UU
    executable, for a SINGLE square 2D binary image. No Mconfig.txt/output files, no
    subprocess call, and no image-size-specific compiled executable are needed - this
    works directly on the numpy array, for any square image size.

    Parameters
    ----------
    image : 2D square binary array (0 = solid/background, 1 = pore/foreground).
    polytopes : which functions to compute. Any of 's2', 'L', 'p3h', 'p3v', 'p4', 'p6'.

    Returns
    -------
    raw : dict {name -> Nt x 2 array of [r, value]}, same layout as np.loadtxt('SobjXXX.txt')
          in the old cpp-based workflow.
    scaled : dict {name -> Nt x 2 array of [r, f(r)]}, the normalised companion curve - same
          as the polyfn_* / scal_autocov attributes on the old Microstructure class. Note: as
          in the original code, 'L' does not get its own scaling formula - it reuses the
          S2-based scaled autocovariance (calculate_polytopes()'s 'L' branch does the same).
    """
    config, maxx, nt = _build_padded_config(image)
    vol_fraction = float(np.mean(image != 0))

    # S2 is always computed: needed on its own if requested, and as the basis of the scaled
    # companion curve for 'L' (mirrors the original code's behaviour, see docstring above).
    s2 = compute_s2_polytope(config, maxx, nt)
    s2_scaled = scale_s2(s2, vol_fraction)

    r_axis = np.arange(nt, dtype=np.float64)
    raw: Dict[str, np.ndarray] = {}
    scaled: Dict[str, np.ndarray] = {}

    for name in polytopes:
        if name == 's2':
            values, values_scaled = s2, s2_scaled
        elif name == 'L':
            values = compute_L_polytope(config, maxx, nt)
            values_scaled = s2_scaled
        elif name == 'p3h':
            values = compute_p3h_polytope(config, maxx, nt)
            values_scaled = scale_polytope_fn(values)
        elif name == 'p3v':
            values = compute_p3v_polytope(config, maxx, nt)
            values_scaled = scale_polytope_fn(values)
        elif name == 'p4':
            values = compute_p4_polytope(config, maxx, nt)
            values_scaled = scale_polytope_fn(values)
        elif name == 'p6':
            values = compute_p6_polytope(config, maxx, nt)
            values_scaled = scale_polytope_fn(values)
        else:
            raise ValueError(f"Unknown polytope '{name}'. Use one of: s2, L, p3h, p3v, p4, p6.")

        raw[name] = np.column_stack((r_axis, values))
        scaled[name] = np.column_stack((r_axis, values_scaled))

    return raw, scaled
