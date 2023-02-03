# install

manage your requirements.txt
```
git@github.com:santex/ship-net.git
cd ship-net;
python3.9 -m pip install -r requirements.txt
```
# skip source data use what is in sat

## c2
bash soho-get.sh c2 "20230101" | xargs wget -rc

## c3

bash soho-get.sh c2 "20230101" | xargs wget -rc

## sdo 1600

bash  bin/sdo-get.sh 1600 2023 01 > 1600-2023-01.txt
cat 1600-2023-01.txt | egrep "(29/|30/)"

## sdo 1700

bash  sdo-get.sh 1700 2023 01  | xargs wget -rc


# run

cd bin;
python3.9 run.py --source ../sat/1600 --clipps ../clipps --sat 1600 --expr 2023



# did it work
check your clipps folder for video & image


![check a 1600 samples](https://github.com/santex/ship-net/blob/master/clipps/1600_smaple.jpg)
![check a 1700 samples](https://github.com/santex/ship-net/blob/master/clipps/1700_smaple.jpg)

# ffmpeg video

a video file should exist
[https://github.com/santex/ship-net/blob/master/clipps/1600_sat.mp4?raw=true]


