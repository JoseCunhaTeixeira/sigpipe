from sigpipe.base.dispersion_curve import VelocityType

CM = 0.3937008  # cm to inch
SINGLE_COLUMN_CM = 9.0
DOUBLE_COLUMN_CM = 18.0
HEIGHT_CM = 9.0
DISP_DPI = 300
SAVING_DPI = 300

VELOCITY_TYPE_LABELS: dict[VelocityType, str] = {
    VelocityType.PHASE: "Phase velocity [m/s]",
    VelocityType.GROUP: "Group velocity [m/s]",
    VelocityType.UNKNOWN: "Velocity [m/s]",
}
