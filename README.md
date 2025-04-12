

```
python simple_augmentations.py --runs_per_vid 5 --batch_size 16
```

copy files TO instance 
```
scp -r * root@server_ip:/root/lerobot_hack_paris/original_videos/
```

copy files FROM finstance
```
scp -r root@server_ip:/root/lerobot_hack_paris/augmented_videos/* ./
```
