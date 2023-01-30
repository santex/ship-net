
start=$(date +"%Y%m%d")
sat=$1
end=$2
while ! [[ $start < $end ]]; do
    #echo $start

for i in $( mojo get "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/2022/$sat/$start/" a attr href | egrep "1024");
do
echo "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/2022/$sat/$start/$i";
done
start=$(date -j -v-1d -f %Y%m%d $start +%Y%m%d)
done
