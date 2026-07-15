import qupath.lib.regions.RegionRequest

def server = getCurrentServer()
if (server == null)
    throw new IllegalStateException('QuPath did not create an image server')

println "SERVER=${server.getClass().getName()}"
println "SIZE=${server.getWidth()}x${server.getHeight()} C=${server.nChannels()} Z=${server.nZSlices()} T=${server.nTimepoints()}"

def request = RegionRequest.createInstance(
    server.getPath(),
    1.0,
    0,
    0,
    Math.min(64, server.getWidth()),
    Math.min(64, server.getHeight())
)
def image = server.readRegion(request)
if (image == null)
    throw new IllegalStateException('QuPath returned no pixels for the smoke-test region')
println "READ=${image.getWidth()}x${image.getHeight()} TYPE=${image.getType()}"
