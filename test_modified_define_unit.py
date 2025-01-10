import unyt

unyt.define_unit("cloudRadius", (1.0, "pc"))
print(unyt.cloudRadius/(2.0 * unyt.pc))

#from unyt import define_unit
#from unyt import pc
#define_unit("cloudRadius", (1.0, "pc"))
#from unyt import cloudRadius
#print(cloudRadius/(2.0*pc))

from importlib import reload
reload(unyt)
# after some processing, switching to a simulation of a larger cloud
unyt.define_unit("cloudRadius", (10.0, "pc"), allow_override=True)
print(unyt.cloudRadius/(2.0 * unyt.pc))

#from unyt import cloudRadius
#print(cloudRadius/(2.0*pc))

