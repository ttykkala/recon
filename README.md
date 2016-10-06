# recon

this package contains a test bench for prototyping with photometric, 

geometric and bi-objective tracking using rgb-d sensor. 

this version is built for 2 CUDA-capable GPUs. 

one doing pose estimation and one reconstruction. 

# installation 

- install cmake 

- install cuda 8.0 

- install qtcreator 

at source directory: 

./build_libs.sh 

mkdir build 

cd build 

ccmake .. -DCMAKE_BUILD_TYPE=Release 

  # configure
  
  # generate 
  
  make 
  
open the project with qtcreator and run it 

cheers, 

tommi m. tykkälä / ttykkala@gmail.com

(also few demo videos at https://www.youtube.com/user/ttykkala9)

# credits 

Roger Sidje for EXPOKIT 

Ramtin Shams for CUDA histogram routines 

Teemu Rantalaiho for CUDA histogram routines 

Tyge Løvset for TinyXML 

Jernej Barbic for performance timer 

Vadim Kutsyy and Jean-Pierre Moreau for cholesky implementation 

# todo 

unify histogram implementation usage 

cleanup 

support for cpu based computation also
