mkdir -p ./data/nuscenes

# nuscenes trainval-set
wget -c -O v1.0-trainval_meta.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
tar -zvxf v1.0-trainval_meta.tgz -C ./data/nuscenes
rm v1.0-trainval_meta.tgz

wget -c -O v1.0-trainval01_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
tar -zvxf v1.0-trainval01_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval01_blobs.tgz
rm v1.0-trainval01_blobs.txt

wget -c -O v1.0-trainval02_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
tar -zvxf v1.0-trainval02_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval02_blobs.tgz
rm v1.0-trainval02_blobs.txt

wget -c -O v1.0-trainval03_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval03_blobs.tgz"
tar -zvxf v1.0-trainval03_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval03_blobs.tgz
rm v1.0-trainval03_blobs.txt

wget -c -O v1.0-trainval04_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
tar -zvxf v1.0-trainval04_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval04_blobs.tgz
rm v1.0-trainval04_blobs.txt

wget -c -O v1.0-trainval05_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval05_blobs.tgz"
tar -zvxf v1.0-trainval05_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval05_blobs.tgz
rm v1.0-trainval05_blobs.txt

wget -c -O v1.0-trainval06_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval06_blobs.tgz"
tar -zvxf v1.0-trainval06_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval06_blobs.tgz
rm v1.0-trainval06_blobs.txt

wget -c -O v1.0-trainval07_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval07_blobs.tgz"
tar -zvxf v1.0-trainval07_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval07_blobs.tgz
rm v1.0-trainval07_blobs.txt

wget -c -O v1.0-trainval08_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"
tar -zvxf v1.0-trainval08_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval08_blobs.tgz
rm v1.0-trainval08_blobs.txt

wget -c -O v1.0-trainval09_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"
tar -zvxf v1.0-trainval09_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval09_blobs.tgz
rm v1.0-trainval09_blobs.txt

wget -c -O v1.0-trainval10_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"
tar -zvxf v1.0-trainval10_blobs.tgz -C ./data/nuscenes
rm v1.0-trainval10_blobs.tgz
rm v1.0-trainval10_blobs.txt

# nuscenes test-set
wget -c -O v1.0-test_meta.tgz "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"
tar -zvxf v1.0-test_meta.tgz -C ./data/nuscenes
rm v1.0-test_meta.tgz

wget -c -O v1.0-test_blobs.tgz "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"
tar -zvxf v1.0-test_blobs.tgz -C ./data/nuscenes
rm v1.0-test_blobs.tgz
rm v1.0-test_blobs.txt