# 10G discspace required

space is occupied between sample data
of two space-craft each with two tools
```
soho c2 c3
sdo  1600 1700
```

# have a look at results here


lists with most of the rest of source urls 
for the images are also found in lists folder
devided in month.
keep in mind nasa changed urls 3 times
over last few years 



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

## the 1600 tool
![check a 1600 samples](https://raw.githubusercontent.com/santex/ship-net/master/clipps/1600_sample.jpg)

### enhanced
![check a 1600 samples](https://raw.githubusercontent.com/santex/ship-net/master/clipps/1600_sample_enhanced.jpg)

## the 1700 tool
![check a 1700 samples](https://raw.githubusercontent.com/santex/ship-net/master/clipps/1700_sample.jpg)

### enhanced
![check a 1700 samples](https://raw.githubusercontent.com/santex/ship-net/master/clipps/1700_sample_enhanced.jpg)
# ffmpeg video

a video file should exist
[https://github.com/santex/ship-net/blob/master/clipps/1600_sat.mp4?raw=true]


