if [ ! -d logdir  ];then
  mkdir logdir
fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_opal_cuda_support=true

#CUDA_VISIBLE_DEVICES=6,7 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --model pp5 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp5_best.pt \
#--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > ./logdir/pp5_2d.log  2>&1 &

#CUDA_VISIBLE_DEVICES=7 mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
#--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot

#CUDA_VISIBLE_DEVICES=3,7 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
#--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot > ./logdir/pp7_2d.log  2>&1 &

#CUDA_VISIBLE_DEVICES=6,7 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
#--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --exp pp7_val --gpu 6,7 > ./logdir/pp7_val.log  2>&1 &

#CUDA_VISIBLE_DEVICES=0,1,2,3 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7_super --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_super.pth.tar \
#--dir_type alpha --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --exp pp7_super --gpu 0,1,2,3 > ./logdir/pp7_super.log  2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup mpirun -n 4 python plot_surface.py --mpi --cuda --ngpu 4 --model pp7_super --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_super.pth.tar \
--dir_type alpha --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --exp pp7_super2 --gpu 0,1,2,3 > ./logdir/pp7_super2.log  2>&1 &

#tail -f ./logdir/pp7_2d.log