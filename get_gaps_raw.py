import sys
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dubai_reports_and_graphs import get_gaps

now_time = datetime.now()
basedir = sys.argv[1]
end_time = datetime(now_time.year, now_time.month, 1)
start_time = end_time - relativedelta(months=1)
#string to store the gaps
gaps_str = ''

for subdir in os.listdir(basedir):
    p = os.path.join(basedir, subdir)
    for f in glob.glob(os.path.join(p, '*.dat')):
        f_base = os.path.basename(f)
        # CurrentBurst rounds dates in another column
        ix_col = 2 if 'CurrentBurst' in f_base else 0
        df = pd.read_csv(f, na_values='NAN', skiprows=[0,2,3], header=0,
                         index_col=ix_col, parse_dates=True)
        ser = df.isnull().sum(axis=1)
        # median isn't 100% reliable to get ts but there is usually enough data
        #interval = np.diff(ser.index.values).median()
        # TODO: refactor to get rid of pandas here
        date_diffs = pd.Series(np.diff(ser.index.values), ser.index[:-1])
        interval = date_diffs.median()
        gaps = get_gaps(ser, start_time, end_time, interval)
        # is there a simpler way to do this?
        times = gaps.apply(lambda x: x.astype(datetime)
                        .apply(lambda v: v.strftime('%Y-%m-%d %H:%M:%S')))
        gaps_zip = ['%s to %s' % (s, e) for s, e in zip(times.start, times.end)]
        gaps_str += "%s: %s\n" % (f_base, gaps_zip)

f_name = '%d-%02d_missing_data_raw.txt' % (start_time.year, start_time.month)
with open(f_name, 'w') as f:
    f.write(gaps_str)
