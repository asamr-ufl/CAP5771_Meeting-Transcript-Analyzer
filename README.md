# Data Directory

## AMI Meeting Corpus

This project uses the AMI Meeting Corpus for meeting transcript efficiency analysis [1].

### How to Obtain the Data

1. Visit: https://groups.inf.ed.ac.uk/ami/download/
2. Download **AMI manual annotations v1.6.2** (22MB)
3. Extract into this `data/` directory
4. Manual transcripts contain utterances with speaker IDs and timestamps

### Data Structure

After extraction, you'll have XML files organized by meeting IDs:

- `EN2001a.*.words.xml` - Individual speaker channels
- `IS1006d.*.words.xml` - Meeting transcript files
- And more...

### Note

**Data files are NOT tracked in Git** due to size. Each team member downloads independently.
