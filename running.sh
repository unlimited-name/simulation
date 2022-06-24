ti=$(date "+%Y%m%d-%H%M%S")
echo 'initial time:'
echo '${ti}'
nohup python -u simulation.py > out.log 2>&1 &
tf=$(date "+%Y%m%d-%H%M%S")
echo 'initial time:'
echo '${ti}'
echo 'final time:'
echo '${tf}'