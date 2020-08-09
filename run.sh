if [ ! -d logdir  ];then
  mkdir logdir
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_opal_cuda_support=true
CUDA_VISIBLE_DEVICES=7 mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

#CUDA_VISIBLE_DEVICES=7 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
#--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > ./logdir/pp7_2d.log  2>&1 &
#tail -f ./logdir/pp7_2d.log