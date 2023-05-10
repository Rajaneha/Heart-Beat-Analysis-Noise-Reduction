import sqlite3
import os
import password1
# Connect to the database
import mysql.connector

# Establish a connection to the MySQL database
conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password=password1.func(),
  database="pack"
)
#conn = sqlite3.connect('music.db')

# Create a cursor
cursor = conn.cursor()

folder = 'E:\\DAVL\\Package\\Data\\after\\set_b\\'

# Create a table with a BLOB column

def create():
    cursor.execute('CREATE TABLE heartbeats (id TEXT PRIMARY KEY, data BLOB)')

# Load binary music data from a file
def load():
    folder_path = r'E:\DAVL\Package\Data\set_b'  # replace with  folder path

    for file_name in os.listdir(folder_path):
            print(file_name)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                music_data = f.read()
            nam = file_name
            # Insert binary music data into the table
            #cursor.execute('INSERT INTO heartbeats (name,data) VALUES (?,?)', (nam,music_data))
            sql = 'INSERT INTO heartbeats_b (name, data) VALUES (%s, %s)'
            binary_data = music_data
            params = (nam, bytes(binary_data))
            cursor.execute(sql, params)

    # Commit the transaction
    conn.commit()

# Close the cursor and connection'''
def take():

    cursor.execute("SELECT * FROM heartbeats_b;")
    rows = cursor.fetchall()
    #b = open("fileaudiowith.wav",'wb')
    for row in rows:
            nam = folder + row[0]
            b = open(nam,'wb')
            #print(row)
            b.write(row[1])
    b.close()


#create()
#load()
take()
            
            
cursor.close()
conn.close()