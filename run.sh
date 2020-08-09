CUDA_VISIBLE_DEVICES=2,3 mpirun -n 4 python plot_surface.py --mpi --cuda --model pp7 --x=-1:1:51 --y=-1:1:51 --model_file snapshots/pp7_best.pt \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --surf_file pp7
