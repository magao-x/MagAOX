#include <stdio.h>
#include <signal.h>
#include <unistd.h>

struct sigaction siga;

void f(int sig) {
    printf("Caught signal %d\n", sig);
}

// sets f as handler to all the possible signals.
void myfunct(void(*f)(int sig)) {
    siga.sa_handler = f;
    for (int sig = 1; sig <= SIGRTMAX; ++sig) {
    int j;
        // this might return -1 and set errno, but we don't care
        j = sigaction(sig, &siga, NULL);
        if (j) printf("%d<=%d ",j,sig);
    }
    printf("\n");
}

int main() {
    myfunct(f);
    while (1) { pause(); } // wait for signal
    return 0;
}
