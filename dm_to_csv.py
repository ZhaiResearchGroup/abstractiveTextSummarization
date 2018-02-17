# dailymail/cnn dataset to csv conversion utility
import sys, os

REPORT_INTERVAL = 5000

def normalize(line):
    return line.replace('\n', '').replace(' , ', ' ')

def main(pathname, offset=0):
    finished = offset

    # Get the first num_files file names
    files = os.listdir(pathname)
    
    # Boolean flag for summary sentences
    summary_next = False
    
    # Read each file and extract vocab
    for filename in files[offset:]:
        with open('%s/%s' % (pathname, filename), 'r') as story_file:
            story = ''
            summary = ''
            for line in story_file:
                if line == '\n': continue # ignore blank lines
                if line == '@highlight\n': summary_next = True
                else:
                    if summary_next: summary += normalize(line) + '. '
                    else: story += normalize(line)
                    summary_next = False
        with open('out.csv', 'a') as csv_file:
            csv_file.write('%d, %s, %s\n' % (finished, story, summary))
        finished += 1
        if finished % REPORT_INTERVAL == 0:
            print("Finished transcribing %d" % finished)
        with open('_status', 'w') as status_file:
            status_file.write('%d' % finished)

if __name__=="__main__":
    try:
        with open('_status') as status_file:
            offset = int(status_file.read())
    except IOError:
        offset = 0
    if len(sys.argv) == 2:
        main(sys.argv[1], offset)
    else:
        print("Usage: python dm_to_csv.py [path to dailymail directory]")
