#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <string.h>
#include <vector>

#define MAX_BUF 1024
using namespace std;

void write_message(string msg) {
  int fd;
  char const * myfifo = "/tmp/fifopipe";
  /* create the FIFO (named pipe) */
  mkfifo(myfifo, 0666);

  /* write message to the FIFO */
  fd = open(myfifo, O_WRONLY);

  write(fd, msg.c_str(), strlen(msg.c_str())+1);
  close(fd);
  cout<<"before unlink"<<endl;
  /* remove the FIFO */
  unlink(myfifo);
}

vector<string> read_message() {
  int fd;
  char const *myfifo = "/tmp/fifopipe2";
  char buf[MAX_BUF];

  /* open, read, and display the message from the FIFO */
  fd = open(myfifo, O_RDONLY, O_NONBLOCK);
  read(fd, buf, MAX_BUF);
  string s(buf, sizeof(buf));
  close(fd);

  vector<string> tokens;
  const char * delimeter = ",";
  char * cstr = new char[s.length() + 1];
  strcpy(cstr, s.c_str());
  char *pch = strtok(cstr, delimeter);

  while (pch != NULL) {
      /* store the next component of the message in a list of strings */
      tokens.push_back(string(pch));
      pch = strtok(NULL, delimeter);
  }
  delete[] cstr;
  //delete[] pch;

  return tokens;
}

int main() {
  vector<string> command_type;
  bool flag = true;

  while(flag) {
    // wait for handshake
    command_type = read_message();
    cout << command_type[0] << endl;

    if (command_type[0] == "1") {
      //clearWindow();
      cout<< "clearWindow()"<<endl;
      write_message("1");
    }
    else if (command_type[0] == "2") {
      write_message("2");

      vector<string> tokens = read_message(); // [width, height, r, g, b, a]
      //createWindow(stof(tokens[0]), stof(tokens[1]),
                    //stof(tokens[2]), stof(tokens[3]), stof(tokens[4]), stof(tokens[5]));
      cout<< "createWindow()"<<endl;
      write_message("1");
    }
    else if (command_type[0] == "3") {
      write_message("3");

      // get points
      bool points_end = false;
      vector<vector<float>> points;
      vector<float> point;
      while (!points_end) {
        vector<string> tokens = read_message(); // "1.03,2.01,0!"
        point = {stof(tokens[0]), stof(tokens[1])}; // [1.03, 2.01]
        points.push_back(point);
        if (point[point.size() - 2] == '1') {
          points_end = true;
        }
        write_message("1");
      }
      //get color
      vector<string> tokens = read_message(); // [1,1,1,1]
      vector<float> color;
      for (string s : tokens) color.push_back(stof(s));
      // draw line
      //drawLine(points, color);
      cout<< "drawLine()"<<endl;
      write_message("1");
    }
    else if (command_type[0] == "4") {
      write_message("4");

      // get points
      bool points_end = false;
      vector<vector<float>> points;
      vector<float> point;
      while (!points_end) {
        vector<string> tokens = read_message(); // "1.03,2.01,0!"
        point = {stof(tokens[0]), stof(tokens[1])}; // [1.03, 2.01]
        points.push_back(point);
        if (point[point.size() - 2] == '1') {
          points_end = true;
        }
        write_message("1");
      }
      //get color
      vector<string> tokens = read_message(); // [1,1,1,1]
      vector<float> color;
      for (string s : tokens) color.push_back(stof(s));
      // draw line
      //drawPolygon(points, color);
      cout<< "drawPolygon()"<<endl;
      write_message("1");
    }
    else if (command_type[0] == "5") {
      //closeWindow();
      flag = false;
      cout<< "closeWindow()"<<endl;
      write_message("1");
    }

  }
  return 0;
}
