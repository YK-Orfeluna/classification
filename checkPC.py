# -*- coding: utf-8 -*-

import platform
import multiprocessing

os = platform.system()
if os == "Darwin" :
	os = "Mac"

cpu = multiprocessing.cpu_count()

print("OS is '%s'" %os)
print("Number of CPU or Thread is '%d'" %cpu)