all:
	source ./env/bin/activate && python nn.py
$(filename):
	source ./env/bin/activate && python $(filename).py

prof:
	nvprof -o nvprof%p.out --profile-child-processes  --unified-memory-profiling off ./env/Scripts/python.exe ./nn.py

vis_prof:
	nvvp --profile-child-processes  --unified-memory-profiling off ./env/Scripts/python.exe ./nn.py

c_prof:
	python -m cProfile -o output.pstats nn.py gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png
