import loci.formats.in.ZarrReader;
import loci.formats.services.OMEXMLService;
import loci.common.services.ServiceFactory;
import java.nio.file.*;
import java.util.*;

class ValidatePublicZarr {
    public static void main(String[] args) throws Exception {
        loci.common.DebugTools.setRootLevel("ERROR");
        Path folder = Path.of(args[0]);
        List<String> names;
        try (var paths = Files.list(folder)) {
            names = paths.filter(p -> p.getFileName().toString().endsWith(".ome.zarr"))
                         .map(Path::toString).sorted().collect(java.util.stream.Collectors.toList());
        }
        for (String name : names) {
            ZarrReader reader = new ZarrReader();
            reader.setFlattenedResolutions(false);
            var metadata = new ServiceFactory().getInstance(OMEXMLService.class).createOMEXMLMetadata();
            reader.setMetadataStore(metadata);
            reader.setId(name);
            int sx=reader.getSizeX(), sy=reader.getSizeY(), c=reader.getSizeC(), t=reader.getSizeT();
            int resolutions=reader.getResolutionCount();
            if (resolutions < 2) throw new Exception("Missing pyramid: "+name);
            for (int level=0;level<resolutions;level++) {
                reader.setResolution(level);
                int expectedX=(sx+(1<<level)-1)/(1<<level);
                int expectedY=(sy+(1<<level)-1)/(1<<level);
                if (reader.getSizeX()!=expectedX || reader.getSizeY()!=expectedY)
                    throw new Exception("Invalid pyramid dimensions: "+name);
                for (int channel=0;channel<c;channel++) {
                    for (int frame : new int[]{0,t-1}) {
                        int index=reader.getIndex(0,channel,frame);
                        int width=Math.min(23,reader.getSizeX()), height=Math.min(19,reader.getSizeY());
                        byte[] bytes=reader.openBytes(index,reader.getSizeX()-width,reader.getSizeY()-height,width,height);
                        if (bytes.length==0) throw new Exception("No pixels: "+name);
                    }
                }
            }
            System.out.println("PASS\t"+Path.of(name).getFileName()+"\t"+sx+"x"+sy+"\tC="+c+"\tT="+t+"\tlevels="+resolutions);
            reader.close();
        }
        if (names.size()!=21) throw new Exception("Expected 21 images, got "+names.size());
    }
}
