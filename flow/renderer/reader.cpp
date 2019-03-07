#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <string.h>

#define MAX_BUF 1024
using namespace std;

char* strrev(char *str){
    int i = strlen(str)-1,j=0;

    char ch;
    while(i>j)
    {
        ch = str[i];
        str[i]= str[j];
        str[j] = ch;
        i--;
        j++;
    }
    return str;

}


int main()
{
    int fd;
    char const *myfifo = "/tmp/fifopipe";
    char buf[MAX_BUF];

    /* open, read, and display the message from the FIFO */
    fd = open(myfifo, O_RDWR, O_NONBLOCK);
    while (strlen(buf) == 0) {
      //cout<< "buffer" << endl;
      read(fd, buf, MAX_BUF);
    }

    cout<<"Received:"<< buf<<endl;
    cout<<"The reversed string is \n"<<strrev(buf)<<endl;
    close(fd);
    return 0;
}
