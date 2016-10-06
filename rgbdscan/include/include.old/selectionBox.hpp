#ifndef SELECTIONBOX_HPP
#define SELECTIONBOX_HPP

class SelectionBox {
private:
public:
    float origin[3];
    float dim[3];
    float color[3];
    char name[512];
    SelectionBox(const char *boxname, float centerX, float centerY, float centerZ, float extentX, float extentY, float extentZ);
    ~SelectionBox();
};

#endif // SELECTIONBOX_HPP
