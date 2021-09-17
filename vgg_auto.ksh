#! /usr/bin/ksh
for i in {1..5}

do 
	typeset -Z3 i
	echo $i
        python vitiligo_vgg.py $i

done

