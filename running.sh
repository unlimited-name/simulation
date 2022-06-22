ti=$(date "+%Y%m%d-%H%M%S")
nohup python simulation.py led1 &
tf=$(date "+%Y%m%d-%H%M%S")
echo 'initial time:'
echo '${ti}'
echo 'final time:'
echo '${tf}'