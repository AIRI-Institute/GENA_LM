import sys
import pybedtools

if len(sys.argv) < 3:
    raise ValueError("Usage: create_accessible_intervals.py <accessible_regions.bed> <output_file.bed> <files...>")

files = sys.argv[1:-1]
output_file = sys.argv[-1]
print ("Loading accessible regions from file: ", files[0])
accessible_regions = pybedtools.BedTool(files[0]).sort().merge()
if len(files) > 1:
    for file in files[1:]:
        print ("Intersecting accessible regions with predictions from file: ", file)
        accessible_regions = accessible_regions.intersect(
                     pybedtools.BedTool(file).sort().merge()
            )
print ("Saving accessible regions to file: ", output_file)
accessible_regions.saveas(output_file)