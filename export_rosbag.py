import roslib; roslib.load_manifest("all_agents_bridge")
import rospy
import rosbag
# from vehicle_interface.msg import PilotRequest
# from auv_msgs.msg import NavSts
import sys
import os
import argparse
import subprocess
import bisect
import json
import yaml
from datetime import datetime
from math import pi

class TimeCache(object):
    def __init__(self):
        self._cache = {}
        self._sorted_list = None

    def add(self, timestamp, value):
        self._cache[timestamp] = value

    def get_closest(self, timestamp):
        if self._sorted_list is None:
            self._sorted_list = sorted(self._cache.keys())
        closest_ts_list_idx = bisect.bisect_left(self._sorted_list, timestamp) - 1
        try:
            closest_ts = self._sorted_list[closest_ts_list_idx]
        except IndexError:
            print("closest_ts_list_idx idx {0} of len {1}".format(closest_ts_list_idx, len(self._sorted_list)))
            closest_ts = self._sorted_list[-1]
        return self._cache[closest_ts]

    def __len__(self):
        return len(self._cache)


class TimeReplayer(object):
    def __init__(self, start_time, increment):
        assert isinstance(start_time, rospy.Time), "start time must be a rospy.Time object"
        assert isinstance(increment, rospy.Duration), "increment must be a rospy.Time object"
        self._start = start_time
        self._increment = increment
        self._now = start_time

    def next(self):
        self._now += self._increment
        return self._now


HACK_WRAP = False


def check_for_pwd(f):
    if "." in f:
        f = f.replace(".", os.path.abspath(os.path.curdir))
    return f


class ExtractField(object):
    def __init__(self, topic_name, field_path, lmda):
        self.topic = topic_name
        self.field_path = field_path
        self.lmbda = lmda

DELIMITER = ','

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export control requests and nav to csv")
    parser.add_argument("config_path", help="Name of the Json configuration file describing "
                                            "what to extract from the rosbag")
    parser.add_argument("bag", nargs='+', help="Name of the ROS bag to load")
    # parser.add_argument(dest="filename", type=check_for_pwd,
    #                     help="Name of the output csv file ([name].csv)", metavar="FILE")
    parser.add_argument("--time", nargs='+', type=int, default=None, help="Start and end time to export (in seconds)")

    args = parser.parse_args()

    with open(args.config_path) as f:
        json_config = json.load(f)

    fields = {}
    for _key in json_config:
        if json_config[_key]["include"] == "True":
            field = ExtractField(json_config[_key]["topic"],
                                 json_config[_key]["field_path"],
                                 json_config[_key]["lambda"])
            fields[_key] = field

    for bag_name in args.bag:
        print("=================================================")
        print("Starting export of bag: {0}".format(bag_name))
        info_dict = yaml.load(
            subprocess.Popen(['rosbag', 'info', '--yaml', bag_name], stdout=subprocess.PIPE).communicate()[0])

        print(info_dict)
        start_time = rospy.Time(info_dict["start"])
        bag_duration = info_dict["duration"]
        end_time = start_time + rospy.Duration(bag_duration)
        print("ROSBAG: {0}".format(bag_name))
        print("     started at: {0}".format(datetime.utcfromtimestamp(start_time.to_sec())))
        print("     duration  : {0}".format(bag_duration))
        print("     ended time: {0}".format(datetime.utcfromtimestamp(end_time.to_sec())))

        if args.time is not None:
            print("--time option provided so exporting window of the bag")
            new_start_time = start_time + rospy.Duration(args.time[0])
            end_time = start_time + rospy.Duration(args.time[1])
            start_time = new_start_time
            print("New start and end times:")
            print("     started at: {0}".format(datetime.utcfromtimestamp(start_time.to_sec())))
            print("     duration  : {0}".format(end_time - start_time))
            print("     ended time: {0}".format(datetime.utcfromtimestamp(end_time.to_sec())))

        bag = rosbag.Bag(bag_name)
        field_keys = fields.keys()
        topics = [field.topic for field in fields.values()]
        caches = {field_key: TimeCache() for field_key in field_keys}
        fields_topic_key = {field.topic: field for field in fields.values()}
        ids_per_topic = {}
        for id, field in fields.items():
            try:
                ids = ids_per_topic[field.topic]
                ids_per_topic[field.topic].append(id)
            except KeyError:
                ids_per_topic[field.topic] = [id]

        for topic, msg, t in bag.read_messages(topics=topics):

            names = ids_per_topic[topic]
            # TODO - add lambda convertor expression to fields

            for name in names:
                value = eval(fields[name].field_path)
                lmda = eval(fields[name].lmbda)
                if lmda is not None:
                    value = lmda(value)
                caches[name].add(t, value)
        bag.close()

        fields_to_remove = []
        for field_key in field_keys:
            cache_size = len(caches[field_key])
            print("Number of msgs for field name {0}: {1}".format(field_key, cache_size))
            if cache_size == 0:
                fields_to_remove.append(field_key)

        if len(fields_to_remove) > 0:
            print("Removing these fields as no data was found in the bag for them:\n{0}".format(fields_to_remove))
            for k in fields_to_remove:
                field_keys.remove(k)


        increment = 0.1
        increment_duration = rospy.Duration(increment)
        replayer = TimeReplayer(start_time, increment_duration)
        bag_name_without_ext = os.path.splitext(bag_name)[0]
        if args.time is None:
            bag_name_without_ext = "{0}-full".format(bag_name_without_ext)
        else:
            bag_name_without_ext = "{0}-{1}-{2}".format(bag_name_without_ext, args.time[0], args.time[1])
        print("Going to write the to output file: {0}".format("{0}.csv".format(bag_name_without_ext)))
        output_file = open("{0}.csv".format(bag_name_without_ext), "w+")
        field_titles = DELIMITER.join(field_keys)
        header = "#{0}\n".format(field_titles)
        output_file.write(header)
        num_steps = int(round((1.0 / increment) * bag_duration))
        print("num steps needed to cover entire bag: {0}".format(num_steps))
        for s in range(num_steps):
            next_timestamp = replayer.next()
            if next_timestamp > end_time:
                break

            field_values = []
            for field_key in field_keys:
                field_values.append(caches[field_key].get_closest(next_timestamp))
            content = DELIMITER.join([str(v) for v in field_values])
            output_file.write("{0}\n".format(content))

        output_file.close()
        print("=================================================")

