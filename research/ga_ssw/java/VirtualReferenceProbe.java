// Bounded reference-generation probe using original SGN classes; no search or energy calls.
import contruction.*;
import nna.BaseConInfo;
import other.ArcFile;
import other.FileHandler;
import java.util.*;

public class VirtualReferenceProbe {
    public static void main(String[] args) throws Exception {
        Model m=ArcFile.getAllModelByArcLowMemory(args[0]).get(0);
        List<ConInfo> refs=BaseConInfo.getVirtualBaseConInfo2(m.getAtoCoos());
        FileHandler.writeConInfos(refs,args[1]);
        System.out.println("Wrote "+refs.size()+" original virtual references");
    }
}
