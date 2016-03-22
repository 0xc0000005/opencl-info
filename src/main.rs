extern crate libc;
use libc::{c_void, size_t};

use std::ptr;
use std::str;
use std::mem;

mod opencl;
use opencl::*;


macro_rules! opencl_get_dev_info {
    ($val_type:ty, $id:expr, $param:expr) => {
        {
            let mut value = 0 as $val_type;
            let error = unsafe {
                clGetDeviceInfo(*$id,
                                $param,
                                mem::size_of::<$val_type>() as size_t,
                                &mut value as *mut _ as *mut c_void,
                                ptr::null_mut())
            };
            assert!(error >= 0, "clGetDeviceInfo()");
            value
        }
    }
}

macro_rules! opencl_get_dev_info_vec {
    ($val_type:ty, $id:expr, $param:expr) => {
        {
            let mut value_size = 0 as size_t;
            let error = unsafe { clGetDeviceInfo(*$id, $param, 0, ptr::null_mut(), &mut value_size) };
            assert!(error >= 0, "clGetDeviceInfo()");

            let mut value: Vec<$val_type> = vec![0 as $val_type; (value_size / mem::size_of::<$val_type>())];
            let error = unsafe {
                clGetDeviceInfo(*$id,
                                $param,
                                value_size,
                                value.as_mut_ptr() as *mut libc::c_void,
                                ptr::null_mut())
            };
            assert!(error >= 0, "clGetDeviceInfo()");
            value
        }
    }
}

fn get_dev_info_string(id: &cl_device_id, param: cl_device_info) -> String {
    let value = opencl_get_dev_info_vec!(u8, id, param);
    String::from(str::from_utf8(&value).unwrap_or("<failed to decode>"))
}

fn get_dev_info_sizes(id: &cl_device_id, param: cl_device_info) -> String {
    let mut result = String::from("");
    for value in opencl_get_dev_info_vec!(size_t, id, param) {
        result = result + &value.to_string() + " ";
    }
    result
}

fn get_platfrom_ids() -> Vec<cl_platform_id> {
    let mut num_of_platforms = 0 as cl_uint;
    let error = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut num_of_platforms) };
    assert!(error >= 0, "clGetPlatformIDs()");

    let mut platforms: Vec<cl_platform_id> =
        vec![0 as cl_platform_id; num_of_platforms as usize];
    let error = unsafe {
        clGetPlatformIDs(num_of_platforms, platforms.as_mut_ptr(), ptr::null_mut())
    };
    assert!(error >= 0, "clGetPlatformIDs()");

    platforms
}

fn print_platfrom_info(id: &cl_platform_id) {
    let param_names = vec![(CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE"),
                           (CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION"),
                           (CL_PLATFORM_NAME, "CL_PLATFORM_NAME"),
                           (CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR"),
                           (CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS")];

    for param in param_names {
        let mut value_size = 0 as size_t;
        let error = unsafe {
            clGetPlatformInfo(*id,
                              param.0 as cl_platform_info,
                              0,
                              ptr::null_mut(),
                              &mut value_size)
        };
        assert!(error >= 0, "clGetPlatformInfo()");

        let mut param_value: Vec<u8> = vec![32u8; value_size as usize];
        let error = unsafe {
            clGetPlatformInfo(*id,
                              param.0 as cl_platform_info,
                              value_size,
                              param_value.as_mut_ptr() as *mut c_void,
                              ptr::null_mut())
        };
        assert!(error >= 0, "clGetPlatformInfo()");

        println!("{} : {}",
                 param.1,
                 str::from_utf8(&param_value).unwrap_or("<failed to decode>"));
    }
}

fn get_devices(id: &cl_platform_id, dev_type: cl_device_type) -> Vec<cl_device_id> {
    let mut num_of_devices = 0 as cl_uint;
    let error = unsafe { clGetDeviceIDs(*id, dev_type, 0, ptr::null_mut(), &mut num_of_devices) };
    assert!(error >= 0, "clGetDeviceIDs()");

    let mut devices: Vec<cl_device_id> = vec![0 as cl_device_id; num_of_devices as usize];
    let error = unsafe {
        clGetDeviceIDs(*id,
                       dev_type,
                       num_of_devices,
                       devices.as_mut_ptr(),
                       ptr::null_mut())
    };
    assert!(error >= 0, "clGetDeviceIDs()");

    devices
}

fn print_device_info(id: &cl_device_id) {
    let dev_type = opencl_get_dev_info!(cl_device_type, id, CL_DEVICE_TYPE);

    print!("Device type :");
    if dev_type & CL_DEVICE_TYPE_DEFAULT != 0 {
        print!(" CL_DEVICE_TYPE_DEFAULT");
    }
    if dev_type & CL_DEVICE_TYPE_CPU != 0 {
        print!(" CL_DEVICE_TYPE_CPU");
    }
    if dev_type & CL_DEVICE_TYPE_GPU != 0 {
        print!(" CL_DEVICE_TYPE_GPU");
    }
    if dev_type & CL_DEVICE_TYPE_ACCELERATOR != 0 {
        print!(" CL_DEVICE_TYPE_ACCELERATOR");
    }
    println!("");

    println!("Vendor : {}", get_dev_info_string(id, CL_DEVICE_VENDOR));
    println!("Name : {}", get_dev_info_string(id, CL_DEVICE_NAME));
    println!("Maximum configured clock frequency (MHz) : {}",
             opencl_get_dev_info!(cl_uint, id, CL_DEVICE_MAX_CLOCK_FREQUENCY));
    println!("Size of global device memory (MB) : {}",
             opencl_get_dev_info!(cl_ulong, id, CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024));
    println!("Size of local memory arena in bytes : {}",
             opencl_get_dev_info!(cl_ulong, id, CL_DEVICE_LOCAL_MEM_SIZE));
    println!("Max size of memory object allocation (MB) : {}",
             opencl_get_dev_info!(cl_ulong, id, CL_DEVICE_MAX_MEM_ALLOC_SIZE) / (1024 * 1024));
    println!("Maximum work item dimensions : {}",
             opencl_get_dev_info!(cl_uint, id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
    println!("Maximum number of work-items that can be specified in each dimension : {}",
             get_dev_info_sizes(id, CL_DEVICE_MAX_WORK_ITEM_SIZES));
    println!("Maximum number of work-items in a work-group : {}",
             get_dev_info_sizes(id, CL_DEVICE_MAX_WORK_GROUP_SIZE));
    println!("Available : {}",
             opencl_get_dev_info!(cl_bool, id, CL_DEVICE_AVAILABLE) != 0);
    println!("Compiler Available : {}",
             opencl_get_dev_info!(cl_bool, id, CL_DEVICE_COMPILER_AVAILABLE) != 0);
    println!("Version : {}", get_dev_info_string(id, CL_DEVICE_VERSION));
    println!("Driver version : {}",
             get_dev_info_string(id, CL_DRIVER_VERSION));
    println!("Profile : {}", get_dev_info_string(id, CL_DEVICE_PROFILE));
    println!("The number of parallel compute cores : {}",
             opencl_get_dev_info!(cl_uint, id, CL_DEVICE_MAX_COMPUTE_UNITS));
    println!("Little Endian : {}",
             opencl_get_dev_info!(cl_bool, id, CL_DEVICE_ENDIAN_LITTLE) != 0);
    println!("Error correction : {}",
             opencl_get_dev_info!(cl_bool, id, CL_DEVICE_ERROR_CORRECTION_SUPPORT) != 0);
    println!("Extensions : {}",
             get_dev_info_string(id, CL_DEVICE_EXTENSIONS));

    println!("");
}

fn main() {
    let platform_ids = get_platfrom_ids();
    for id in platform_ids {
        print_platfrom_info(&id);
        let devices = get_devices(&id, CL_DEVICE_TYPE_ALL);
        println!("\nNumber of detected OpenCL devices : {}\n", devices.len());
        for dev in devices {
            print_device_info(&dev);
        }
    }
}
