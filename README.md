# get source data to put in sat

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

![check and find similar](https://github.com/santex/ship-net/blob/master/clipps/20230129/133103_008080c0c0c0c080_1_899_357_240_234.jpg)
