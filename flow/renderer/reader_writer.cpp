#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
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
// write
    int fd;
    char const * myfifo = "/tmp/fifopipe";
    /* create the FIFO (named pipe) */
    mkfifo(myfifo, 0666);

    /* write message to the FIFO */
    fd = open(myfifo, O_WRONLY);

    char const *msg;
    msg="This is the string to be reversed";
    write(fd, msg, strlen(msg)+1);
    close(fd);
    cout<<"before unlink"<<endl;
    /* remove the FIFO */
    unlink(myfifo);


    return 0;
}
