
# coding: utf-8

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
import errno
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt
from credentials import dsn

import matplotlib.dates as mpd
from collections import OrderedDict


def pivot_sensor_data(df, group_name, sensor_lookup):
    remade = df.merge(sensor_lookup, how='left', on='sensor_id').drop('sensor_id', axis=1)
    # if there's an empty dataframe, return nothing
    if remade.empty:
        return None, None, None
    else:
        bin_str = (' / Bin #' + (remade.bin + 1).fillna('').astype(str).str.extract(r'^(\d+)')).fillna('')
        fill_col = remade.depth.fillna('').astype(str).str.replace(r'(\d+\.?\d*)', r' at \1 m')
        # show the units, along with depth and bin if applicable
        remade.unit = remade.unit + fill_col + bin_str
        remade.drop('depth', axis=1, inplace=True)
        pivoted = pd.tools.pivot.pivot_table(remade, index='time', columns='unit').sort()
        # round down to this many minutes expressed as nanoseconds
        date_diffs = pd.Series(np.diff(pivoted.index.values), pivoted.index[:-1])
        interval = date_diffs.median()
        try:
            # nanoseconds
            round_to = interval.total_seconds() * 1e9
        except ZeroDivisionError:
            print(group_name + 'failed due to zero division error.')
            return None, None, None
        round_idx = pd.DatetimeIndex((pivoted.index.astype(np.int64) // round_to) * round_to)
        round_idx
        pivoted['Original TIME (+04:00)'] = pivoted.index.values
        pivoted.index = round_idx
        vals = pivoted['value']
        pivoted.index.name = 'Time'
        vals_masked = vals.copy()
        pivoted.name = group_name
        return (pivoted, vals_masked, interval)


def get_gaps(ser, start_time, end_time, interval):
        # the pivot operation filled in missing vals with NaNs, remove any
        # from series before we begin
        ser_no_na = ser.dropna()
        #set gaps for start/end appropriately
        # empty series will screw up the calculation, so handle them differently
        if not ser_no_na.empty:
            #TODO: highlight gaps at beginning/end of file
            date_diffs = pd.Series(np.diff(ser_no_na.index.values), ser_no_na.index[:-1])
            # perhaps not DRY enough
            interval = date_diffs.median()
            ser_no_na = ser.dropna().combine_first(pd.Series(index=[start_time, end_time - interval]))
            date_diffs = pd.Series(np.diff(ser_no_na.index.values), ser_no_na.index[:-1])
            # give reasonable tolerances for timestamp offsets
            try:
                time_sec = interval.total_seconds()
            except Exception:
                pass
            time_pred = ((date_diffs > timedelta(seconds=0.8 * time_sec)) &
                         (date_diffs > timedelta(seconds=1.2 * time_sec)))
            gaps = date_diffs[time_pred]
            gap_times = (gaps.index.values + gaps).reset_index()
            gap_times.columns = ['start', 'end']

            return gap_times
        else:
            return pd.DataFrame(columns=['start', 'end'])


def plot_quality(ri_vals, qual_mask, title, start_time, end_time, path):
    # replace with empty df rather than using None?
    if ri_vals is None:
        return
    cmap = ListedColormap(['r', 'b'])
    # use this to make sure colors are set properly for True, False vals
    norm = Normalize(False, True)

    for name in [col for col in ri_vals.columns if col != 'Original TIME (+04:00)']:
        # x should stay the same
        x = mpd.date2num(ri_vals[name].index.to_pydatetime())
        y = ri_vals[name].values
        # highlight around point
        z = np.append(qual_mask.ix[1:][name].values, True) & qual_mask[name].values
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(z)
        fig1 = plt.figure(figsize=(20,5))
        plt.hold(True)
        ax = plt.gca()
        inv_idx = (qual_mask[name] == False).values
        # FIXME: use either DF/Series or numpy arrays, not both
        invalid = ri_vals[name][qual_mask[name] == False].values
        invalid_x = x[inv_idx]
        ax.add_collection(lc)
        # get only bad points for scatter plot overlay
        ax.scatter(invalid_x, invalid, c='r')
        ax.xaxis.set_major_formatter(mpd.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mpd.DayLocator())
        ax.set_xlabel('Date')
        ax.set_ylabel(ri_vals[name].name)
        plt.suptitle(title)
        try:
            #_=plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        except Exception as e:
            print(e)
            return
        # for some reason autofitting does not occur, fit to to first and last of time series
        # use axis='x' arg if only x is desired
        # show gaps with no data as vertical spans
        # TODO: refactor to put outside of plot_quality.  Have plot quality
        # handle a single parameter at a time instead of iterating over many
        # parameters
        gaps = get_gaps(ri_vals[name], start_time, end_time, interval)
        for start, end in gaps.values:
            plt.axvspan(mpd.date2num(pd.to_datetime(start)), mpd.date2num(pd.to_datetime(end)),
                        edgecolor='none', facecolor='#ff9999')
        ax.grid(which='major')
        plt.xlim(x[0], x[-1])
        # fit plot bounds for figure export
        # FIXME: plot title can partially overlap plot area when
        fig1.tight_layout()
        # name ADCPs according to bin number
        if 'Bin' in name:
            fname_start = re.sub(r' \([^)]*\).*/ Bin #(\d+)$', r' bin_\1', name)
        else:
            fname_start = re.sub(r' \([^)]*\)$', r'', name)
        fname = title + '-' + fname_start.lower().replace(' ', '_') + '.png'
        plt.savefig(os.path.join(path, fname))
        plt.close(fig1)
        print(fname + ' was written')

def reindex_vals(pivoted, interval, start, end):

    # drop any duplicate records prior to loading
    no_dup = pivoted.reset_index().drop_duplicates('Time').set_index('Time')
    # FIXME: get rid of hardcoding
    resamp_freq = str(int(interval.total_seconds() / 60)) + 'Min'
    dates = pd.DatetimeIndex(pd.date_range(start, end, freq=resamp_freq,
                                           closed='left'))
    ri = no_dup.reindex(dates)
    # if axes are exactly the same, use the initial axes
    ri_vals = ri['value']
    qual_filled = ri['quality_id'].fillna(True)
    qual_mask = (qual_filled >= 1) & (qual_filled <= 2)
    orig_time = ri['Original TIME (+04:00)']
    orig_time.name = 'Original TIME (+04:00)'
    return ri_vals, qual_mask, orig_time


def calc_stats(ri_vals, pivoted):
    # TODO: drop some columns based on whether or not they need to be resampled
    stats = pd.DataFrame(OrderedDict([('Count', ri_vals.count()),
                              ('Expected', ri_vals.fillna(0).count()),
                              ('QC Count', ((pivoted['quality_id'] < 1) | (pivoted['quality_id'] > 2)).sum())]))
    stats['Percentage'] = stats['Count'] / stats['Expected'] * 100

    return stats


engine = create_engine(dsn, encoding='latin1')

# get proper units/abbrevs
sensor_lookup = pd.read_sql("""SELECT s.id sensor_id, CONCAT(p.readable,
                               COALESCE(CASE WHEN display_unit IS NOT NULL THEN CONCAT(' (', display_unit, ')') ELSE display_unit END,
                                        CASE WHEN abbreviation = 'None' THEN ''
                                        ELSE CONCAT(' (', abbreviation, ')') END)) AS unit
                               FROM sensors s JOIN parameters p ON s.parameter_id = p.id JOIN units u ON p.unit_id = u.id""",
                               engine)
sensor_lookup.unit = sensor_lookup.unit.map(lambda s: unicode(s, encoding='latin1'))
now_time = datetime.now()
end_time = datetime(now_time.year, now_time.month, 1)
start_time = end_time - relativedelta(months=1)
# look for active stations
st_groups = pd.read_sql("""SELECT s.id, s.code from stations s JOIN
                         (select distinct station_id from groups_stations) t
                         ON t.station_id = s.id WHERE s.active AND s.exposed""", engine)
for (id, station) in st_groups.to_records(index=False):
    # get sensor groups for each active station
    s_groups = pd.read_sql("""SELECT group_name FROM groups g
                                 JOIN groups_stations gs ON g.id = gs.group_id
                                 WHERE gs.station_id = %s
                                 AND group_name NOT LIKE '%%ErrorEcho'""", engine, params=(id,))
    print(s_groups)
    sg_vals = []
    stat_arr = []
    for (grp,) in s_groups.to_records(index=False):
        path = os.path.join(start_time.strftime('%Y-%m'), grp)
        # Python 3 has os.makedirs(dir, exist_ok=True) for `mkdir -p` like
        # functionality, but we want to go for cross-version support
        print(grp)
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        # check timestamps!  rails stores in UTC since this mysql config is not TZ aware
        df = pd.read_sql("""SELECT CONVERT_TZ(o.time, 'UTC', 'Asia/Dubai') time, o.sensor_id, o.quality_id, o.value, s.bin, s.bin * v.bin_size + v.value AS depth
                                        from observations o
                                        JOIN groups_sensors gs ON o.sensor_id = gs.sensor_id
                                        JOIN groups g ON gs.group_id = g.id
                                        JOIN sensors s ON s.id = gs.sensor_id
                                        LEFT JOIN variations v ON v.id = s.variation_id
                              WHERE g.group_name = %s AND o.time >= CONVERT_TZ(%s, 'Asia/Dubai', 'UTC')
                                     AND o.time < CONVERT_TZ(%s, 'Asia/Dubai', 'UTC')""", engine, params=(grp, start_time.isoformat(),
                                                                                                            end_time.isoformat()))

        pivoted, val, interval = pivot_sensor_data(df, grp, sensor_lookup)
        if pivoted is not None:
            ri_val, qual_mask, orig_time = reindex_vals(pivoted, interval, start=start_time,
                                                      end=end_time)
            # NaN out QCed values
            vals_masked = ri_val.mask(~qual_mask.fillna(False))
            vals_masked.columns = vals_masked.columns.get_values()
            time_merge = pd.concat([orig_time, vals_masked], axis=1)
            time_merge.index.name = 'Formatted TIME (+04:00)'
            # can probably use astype(str) instead
            replace_time = time_merge['Original TIME (+04:00)'].apply(lambda d: str(d)).str.replace('NaT', 'None found')
            time_merge['Original TIME (+04:00)'] = replace_time
            time_merge.to_csv(grp + '.csv', encoding='utf-8', na_rep='NAN')
            # however, show QCed values on plot
            plot_quality(ri_val, ri_qual, grp, start_time, end_time, path)
            stat_arr.append(calc_stats(ri_val, pivoted))

    if stat_arr:
        sensor_stats = pd.concat(stat_arr)
        sensor_stats.index.name = 'Parameter'
        print('Writing station ' + station + ' to csv')
        sensor_stats.to_csv(station + '.csv', encoding='utf-8', na_rep='NAN')
    else:
        print('Station %s had no data for this time period' % station)
