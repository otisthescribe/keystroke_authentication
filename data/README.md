# Data description

free_text_data.pickle contains the concatenated data from three other files in this folder.
Files were merged based on the UserID to expend the data from each user
to the most possible extent.

Each file is a pickle file containing a python dictionary. Data should be sorted based on the
length of the user's data.

A free_text_data.pickle contains dictionary with this structure:
```
{
    0: [hold time, between time, ..., hold, between]
    1: [hold time, between time, ..., hold, between]
    ...
    N: [hold time, between time, ..., hold, between]
}
```

The number of users is around 1000 after merging three files. The number of users in
certain file is as follows:
- GayMarriage.pickle = 400 users
- GunControl.pickle = 400 users
- ReviewAMT.pickle = 500 users