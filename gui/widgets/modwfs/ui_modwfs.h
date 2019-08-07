/********************************************************************************
** Form generated from reading UI file 'modwfs.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MODWFS_H
#define UI_MODWFS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_modwfs
{
public:
    QGridLayout *gridLayout;
    QLabel *titleLabel;
    QLabel *ttmStatus;
    QGridLayout *gridLayout_2;
    QPushButton *buttonRest;
    QPushButton *buttonSet;
    QPushButton *buttonModulate;
    QLabel *labelChannel1;
    QLCDNumber *voltsAxis1;
    QPushButton *buttonUp;
    QPushButton *buttonLeft;
    QPushButton *buttonRight;
    QLabel *labelChannel2;
    QLCDNumber *voltsAxis2;
    QPushButton *buttonDown;
    QPushButton *pushButton;

    void setupUi(QWidget *modwfs)
    {
        if (modwfs->objectName().isEmpty())
            modwfs->setObjectName(QStringLiteral("modwfs"));
        modwfs->resize(551, 272);
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(modwfs->sizePolicy().hasHeightForWidth());
        modwfs->setSizePolicy(sizePolicy);
        modwfs->setSizeIncrement(QSize(1, 1));
        QPalette palette;
        QBrush brush(QColor(32, 31, 31, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush);
        QBrush brush1(QColor(0, 0, 0, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush1);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush1);
        modwfs->setPalette(palette);
        modwfs->setFocusPolicy(Qt::StrongFocus);
        gridLayout = new QGridLayout(modwfs);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        titleLabel = new QLabel(modwfs);
        titleLabel->setObjectName(QStringLiteral("titleLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(titleLabel->sizePolicy().hasHeightForWidth());
        titleLabel->setSizePolicy(sizePolicy1);
        titleLabel->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(titleLabel, 0, 0, 1, 5);

        ttmStatus = new QLabel(modwfs);
        ttmStatus->setObjectName(QStringLiteral("ttmStatus"));
        sizePolicy1.setHeightForWidth(ttmStatus->sizePolicy().hasHeightForWidth());
        ttmStatus->setSizePolicy(sizePolicy1);
        QPalette palette1;
        QBrush brush2(QColor(170, 255, 255, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette1.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        QBrush brush3(QColor(96, 95, 94, 255));
        brush3.setStyle(Qt::SolidPattern);
        palette1.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        ttmStatus->setPalette(palette1);
        QFont font;
        font.setFamily(QStringLiteral("Tlwg Typist"));
        font.setPointSize(22);
        ttmStatus->setFont(font);
        ttmStatus->setFrameShape(QFrame::Box);
        ttmStatus->setFrameShadow(QFrame::Sunken);
        ttmStatus->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(ttmStatus, 1, 0, 1, 5);

        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setSizeConstraint(QLayout::SetMinimumSize);
        buttonRest = new QPushButton(modwfs);
        buttonRest->setObjectName(QStringLiteral("buttonRest"));
        sizePolicy.setHeightForWidth(buttonRest->sizePolicy().hasHeightForWidth());
        buttonRest->setSizePolicy(sizePolicy);
        buttonRest->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonRest, 0, 0, 1, 1);

        buttonSet = new QPushButton(modwfs);
        buttonSet->setObjectName(QStringLiteral("buttonSet"));
        sizePolicy.setHeightForWidth(buttonSet->sizePolicy().hasHeightForWidth());
        buttonSet->setSizePolicy(sizePolicy);
        buttonSet->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonSet, 0, 1, 1, 1);

        buttonModulate = new QPushButton(modwfs);
        buttonModulate->setObjectName(QStringLiteral("buttonModulate"));
        sizePolicy.setHeightForWidth(buttonModulate->sizePolicy().hasHeightForWidth());
        buttonModulate->setSizePolicy(sizePolicy);
        buttonModulate->setFocusPolicy(Qt::StrongFocus);

        gridLayout_2->addWidget(buttonModulate, 0, 2, 1, 1);

        gridLayout_2->setColumnStretch(0, 1);
        gridLayout_2->setColumnStretch(1, 1);
        gridLayout_2->setColumnStretch(2, 1);

        gridLayout->addLayout(gridLayout_2, 2, 0, 1, 5);

        labelChannel1 = new QLabel(modwfs);
        labelChannel1->setObjectName(QStringLiteral("labelChannel1"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(2);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(labelChannel1->sizePolicy().hasHeightForWidth());
        labelChannel1->setSizePolicy(sizePolicy2);
        QPalette palette2;
        QBrush brush4(QColor(0, 0, 102, 255));
        brush4.setStyle(Qt::SolidPattern);
        palette2.setBrush(QPalette::Active, QPalette::Text, brush4);
        palette2.setBrush(QPalette::Inactive, QPalette::Text, brush4);
        QBrush brush5(QColor(83, 82, 81, 255));
        brush5.setStyle(Qt::SolidPattern);
        palette2.setBrush(QPalette::Disabled, QPalette::Text, brush5);
        labelChannel1->setPalette(palette2);
        QFont font1;
        font1.setFamily(QStringLiteral("Tlwg Typewriter"));
        font1.setPointSize(14);
        labelChannel1->setFont(font1);

        gridLayout->addWidget(labelChannel1, 3, 0, 2, 1);

        voltsAxis1 = new QLCDNumber(modwfs);
        voltsAxis1->setObjectName(QStringLiteral("voltsAxis1"));
        QSizePolicy sizePolicy3(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(2);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(voltsAxis1->sizePolicy().hasHeightForWidth());
        voltsAxis1->setSizePolicy(sizePolicy3);
        QPalette palette3;
        palette3.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette3.setBrush(QPalette::Active, QPalette::Text, brush2);
        palette3.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette3.setBrush(QPalette::Inactive, QPalette::Text, brush2);
        palette3.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        palette3.setBrush(QPalette::Disabled, QPalette::Text, brush5);
        voltsAxis1->setPalette(palette3);

        gridLayout->addWidget(voltsAxis1, 3, 1, 2, 1);

        buttonUp = new QPushButton(modwfs);
        buttonUp->setObjectName(QStringLiteral("buttonUp"));
        QSizePolicy sizePolicy4(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(1);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(buttonUp->sizePolicy().hasHeightForWidth());
        buttonUp->setSizePolicy(sizePolicy4);
        buttonUp->setMaximumSize(QSize(50, 16777215));
        buttonUp->setFocusPolicy(Qt::StrongFocus);

        gridLayout->addWidget(buttonUp, 3, 3, 1, 1);

        buttonLeft = new QPushButton(modwfs);
        buttonLeft->setObjectName(QStringLiteral("buttonLeft"));
        sizePolicy4.setHeightForWidth(buttonLeft->sizePolicy().hasHeightForWidth());
        buttonLeft->setSizePolicy(sizePolicy4);
        buttonLeft->setMaximumSize(QSize(50, 16777215));
        buttonLeft->setFocusPolicy(Qt::StrongFocus);

        gridLayout->addWidget(buttonLeft, 4, 2, 2, 1);

        buttonRight = new QPushButton(modwfs);
        buttonRight->setObjectName(QStringLiteral("buttonRight"));
        sizePolicy4.setHeightForWidth(buttonRight->sizePolicy().hasHeightForWidth());
        buttonRight->setSizePolicy(sizePolicy4);
        buttonRight->setMaximumSize(QSize(50, 16777215));
        buttonRight->setFocusPolicy(Qt::StrongFocus);

        gridLayout->addWidget(buttonRight, 4, 4, 2, 1);

        labelChannel2 = new QLabel(modwfs);
        labelChannel2->setObjectName(QStringLiteral("labelChannel2"));
        sizePolicy2.setHeightForWidth(labelChannel2->sizePolicy().hasHeightForWidth());
        labelChannel2->setSizePolicy(sizePolicy2);
        labelChannel2->setFont(font1);

        gridLayout->addWidget(labelChannel2, 5, 0, 2, 1);

        voltsAxis2 = new QLCDNumber(modwfs);
        voltsAxis2->setObjectName(QStringLiteral("voltsAxis2"));
        sizePolicy3.setHeightForWidth(voltsAxis2->sizePolicy().hasHeightForWidth());
        voltsAxis2->setSizePolicy(sizePolicy3);
        QPalette palette4;
        palette4.setBrush(QPalette::Active, QPalette::WindowText, brush2);
        palette4.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
        palette4.setBrush(QPalette::Disabled, QPalette::WindowText, brush3);
        voltsAxis2->setPalette(palette4);

        gridLayout->addWidget(voltsAxis2, 5, 1, 2, 1);

        buttonDown = new QPushButton(modwfs);
        buttonDown->setObjectName(QStringLiteral("buttonDown"));
        sizePolicy4.setHeightForWidth(buttonDown->sizePolicy().hasHeightForWidth());
        buttonDown->setSizePolicy(sizePolicy4);
        buttonDown->setMaximumSize(QSize(50, 16777215));
        buttonDown->setFocusPolicy(Qt::StrongFocus);

        gridLayout->addWidget(buttonDown, 6, 3, 1, 1);

        pushButton = new QPushButton(modwfs);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        sizePolicy4.setHeightForWidth(pushButton->sizePolicy().hasHeightForWidth());
        pushButton->setSizePolicy(sizePolicy4);
        pushButton->setMaximumSize(QSize(50, 16777215));

        gridLayout->addWidget(pushButton, 5, 3, 1, 1);


        retranslateUi(modwfs);

        QMetaObject::connectSlotsByName(modwfs);
    } // setupUi

    void retranslateUi(QWidget *modwfs)
    {
        modwfs->setWindowTitle(QApplication::translate("modwfs", "modwfs", Q_NULLPTR));
        titleLabel->setText(QApplication::translate("modwfs", "PyWFS Modulator", Q_NULLPTR));
        ttmStatus->setText(QApplication::translate("modwfs", "OFF", Q_NULLPTR));
        buttonRest->setText(QApplication::translate("modwfs", "Rest", Q_NULLPTR));
        buttonSet->setText(QApplication::translate("modwfs", "Set", Q_NULLPTR));
        buttonModulate->setText(QApplication::translate("modwfs", "Modulate", Q_NULLPTR));
        labelChannel1->setText(QApplication::translate("modwfs", "Channel 1:", Q_NULLPTR));
        buttonUp->setText(QApplication::translate("modwfs", "U", Q_NULLPTR));
        buttonLeft->setText(QApplication::translate("modwfs", "L", Q_NULLPTR));
        buttonRight->setText(QApplication::translate("modwfs", "R", Q_NULLPTR));
        labelChannel2->setText(QApplication::translate("modwfs", "Channel 2:", Q_NULLPTR));
        buttonDown->setText(QApplication::translate("modwfs", "D", Q_NULLPTR));
        pushButton->setText(QApplication::translate("modwfs", "0.01", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class modwfs: public Ui_modwfs {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MODWFS_H
