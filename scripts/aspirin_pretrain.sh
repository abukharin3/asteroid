CUDA_VISIBLE_DEVICES=0 python pretrain_sup.py \
--evaluation_interval 200 \
--n_pts 200 \
--num_steps 40000 \
--num_spherical 7 \
--num_radial 6 \
--num_blocks 4 \
--emb_size_atom 128 \
--emb_size_edge 128 \
--molecule aspirin \
--n_vectors 1 \
--batch_size 10 \
--comment pretrain_aspirin_200 \
--model_name pretrain_aspirin_200 \
--ccsd \
--decay_steps 1200000 \
--warmup_steps 10000 \
--decay_patience 50000 \
--decay_cooldown 50000 \
--save_model


CUDA_VISIBLE_DEVICES=0 python get_predictions.py \
--molecule aspirin \
--num_spherical 7 \
--num_radial 6 \
--num_blocks 4 \
--emb_size_atom 128 \
--emb_size_edge 128 \
--n_pts 2000 \
--revised \
--model_name pretrain_aspirin_200EPOCH

CUDA_VISIBLE_DEVICES=0 python pretrain_asteroid.py \
--evaluation_interval 100 \
--num_steps 200000 \
--num_spherical 7 \
--num_radial 6 \
--num_blocks 4 \
--emb_size_atom 128 \
--emb_size_edge 128 \
--molecule aspirin \
--n_vectors 1 \
--batch_size 10 \
--revised \
--comment aspirin_pretrain_200_gamma_0.1_2000 \
--save_model \
--gamma 0.1 \
--model_name pretrain_aspirin_200EPOCH_2000_gamma0.1 \
--load_name pretrain_aspirin_200EPOCH_2000
