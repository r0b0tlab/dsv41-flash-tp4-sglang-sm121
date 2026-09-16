"""Execute the byte-frozen text180 scorer with our four-rank admission transport."""
import hashlib,os,pathlib,runpy,sys
HERE=pathlib.Path(__file__).resolve().parent
KIT=pathlib.Path(os.environ.get('Q200_KIT',pathlib.Path.home()/'projects/r0b0bench/subsets/q200v2'))
sys.path.insert(0,str(HERE/'q200_support'))
sys.argv=[str(KIT/'scripts/run_quality_set.py'),*sys.argv[1:]]
runpy.run_path(sys.argv[0],run_name='__main__')
