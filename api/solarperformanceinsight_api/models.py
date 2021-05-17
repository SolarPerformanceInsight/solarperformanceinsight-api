import datetime as dt
from enum import Enum
from typing import Union, List, Optional, Any, Tuple, Dict


import pandas as pd
import pvlib  # type: ignore
from pvlib.ivtools.sdm import pvsyst_temperature_coeff  # type: ignore
from pydantic import BaseModel, Field, PrivateAttr, validator, root_validator
from pydantic.fields import Undefined
from pydantic.types import UUID
import pytz


SYSTEM_ID = "6b61d9ac-2e89-11eb-be2a-4dc7a6bcd0d9"
SYSTEM_EXAMPLE = dict(
    name="Test PV System",
    latitude=33.98,
    longitude=-115.323,
    elevation=2300,
    inverters=[
        dict(
            name="Inverter 1",
            make_model="ABB__MICRO_0_25_I_OUTD_US_208__208V_",
            inverter_parameters=dict(
                Pso=2.08961,
                Paco=250,
                Pdco=259.589,
                Vdco=40,
                C0=-4.1e-05,
                C1=-9.1e-05,
                C2=0.000494,
                C3=-0.013171,
                Pnt=0.075,
            ),
            losses={},
            arrays=[
                dict(
                    name="Array 1",
                    make_model="Canadian_Solar_Inc__CS5P_220M",
                    albedo=0.2,
                    modules_per_string=7,
                    strings=5,
                    tracking=dict(
                        tilt=20.0,
                        azimuth=180.0,
                    ),
                    temperature_model_parameters=dict(
                        u_c=29.0, u_v=0.0, eta_m=0.1, alpha_absorption=0.9
                    ),
                    module_parameters=dict(
                        alpha_sc=0.004539,
                        gamma_ref=1.2,
                        mu_gamma=-0.003,
                        I_L_ref=5.11426,
                        I_o_ref=8.10251e-10,
                        R_sh_ref=381.254,
                        R_s=1.06602,
                        R_sh_0=400.0,
                        cells_in_series=96,
                    ),
                )
            ],
            airmass_model="kastenyoung1989",
            aoi_model="physical",
            clearsky_model="ineichen",
            spectral_model="no_loss",
            transposition_model="haydavies",
        )
    ],
)
# all compatible with luxon 1.25.0, most commen + Etc/GMT+offset
TIMEZONES = [
    tz
    for tz in pytz.common_timezones
    if not tz.startswith("US/") and tz not in ("America/Nuuk", "Antarctica/McMurdo")
] + [tz for tz in pytz.all_timezones if tz.startswith("Etc/GMT") and tz != "Etc/GMT0"]
SURFACE_ALBEDOS = pvlib.irradiance.SURFACE_ALBEDOS
TEMPERATURE_PARAMETERS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS


# allows word chars, space, comma, apostrophe, hyphen, parentheses, underscore
# and empty string
def UserString(default: Any = Undefined, *, title: str = None, description: str = None):
    return Field(
        default,
        title=title,
        description=description,
        max_length=128,
        regex=r"^(?!\W+$)(?![_ ',\-\(\)]+$)[\w ',\-\(\)]*$",
    )


class SPIBase(BaseModel):
    class Config:
        extra = "forbid"


class PVLibBase(SPIBase):
    """Provide a `pvlib_dict` method to convert parameters if needed
    for using in pvlib. Child classes may implement model-specific conversions
    as needed."""

    def pvlib_dict(self):
        return self.dict()


class FixedTracking(SPIBase):
    """Parameters for a fixed tilt array"""

    tilt: float = Field(
        ..., description="Tilt of modules in degrees from horizontal", ge=0, le=180
    )
    azimuth: float = Field(
        ...,
        description="Azimuth of modules relative to North in degrees",
        ge=0,
        lt=360.0,
    )


class SingleAxisTracking(SPIBase):
    """Parameters for a single axis tracking array"""

    axis_tilt: float = Field(
        ...,
        title="Axis Tilt",
        description="Tilt of tracker axis in degrees from horizontal",
        ge=0,
        le=90,
    )
    axis_azimuth: float = Field(
        ...,
        title="Axis Azimiuth",
        description="Azimuth of tracker axis clockwise from North in degrees",
        ge=0,
        lt=360.0,
    )
    gcr: float = Field(
        ...,
        title="GCR",
        description=(
            "Ground coverage ratio: ratio of module length to the spacing"
            " between trackers"
        ),
        ge=0,
    )
    backtracking: bool = Field(
        ..., description="True if the tracking system supports backtracking"
    )


class PVsystModuleParameters(PVLibBase):
    """Parameters for the modules that make up an array in a PVsyst-like model"""

    alpha_sc: float = Field(
        ...,
        description=(
            "Short-circuit current temperature coefficient of the module "
            "in units of A/C"
        ),
    )
    gamma_ref: float = Field(..., description="Diode ideality factor", ge=0)
    mu_gamma: float = Field(
        ...,
        description="Temperature coefficient for the diode ideality factor, 1/K",
    )
    I_L_ref: float = Field(
        ...,
        description=(
            "Light-generated current (or photocurrent) at reference conditions,"
            "in amperes"
        ),
    )
    I_o_ref: float = Field(
        ...,
        description=(
            "Dark or diode reverse saturation current at reference conditions,"
            "in amperes"
        ),
    )
    R_sh_ref: float = Field(
        ...,
        description="Shunt resistance at reference conditions, in ohms",
    )
    R_sh_0: float = Field(
        ..., description="Shunt resistance at zero irradiance conditions, in ohms"
    )
    R_s: float = Field(
        ..., description="Series resistance at reference conditions, in ohms"
    )
    cells_in_series: int = Field(
        ..., description="Number of cells connected in series in a module", ge=0
    )
    R_sh_exp: float = Field(
        5.5, description="Exponent in the equation for shunt resistance, unitless"
    )
    EgRef: float = Field(
        1.121,
        description=(
            "Energy bandgap at reference temperature in units of eV. "
            "1.121 eV for crystsalline silicon."
        ),
        gt=0,
    )
    _modelchain_dc_model: str = PrivateAttr("pvsyst")
    _gamma: float = PrivateAttr()

    @root_validator(skip_on_failure=True)
    def validate_diode_params(cls, values):
        # and set _gamma here since it could raise an error with bad params
        err = ValueError(
            "Unable to calculate single diode parameters from parameters supplied."
        )
        try:
            pvlib.pvsystem.calcparams_pvsyst(
                effective_irradiance=1000, temp_cell=25, **values
            )
        except Exception:
            raise err
        try:
            cls._gamma = pvsyst_temperature_coeff(**values)
        except Exception:
            raise err
        return values


class PVWattsModuleParameters(PVLibBase):
    """Parameters for the modules that make up an array in a PVWatts-like model"""

    pdc0: float = Field(
        ...,
        description="Power of the modules at 1000 W/m^2 and cell reference temperature",
    )
    gamma_pdc: float = Field(
        ...,
        description=(
            "Temperature coefficient of power in units of %/C. "
            "Typically -0.2 to -0.5 % per degree C"
        ),
    )
    _modelchain_dc_model: str = PrivateAttr("pvwatts")
    _gamma: float = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._gamma = self.gamma_pdc / 100

    def pvlib_dict(self):
        """Convert to a dict pvlib understands for `module_parameters`
        i.e. scale gamma_pdc to 1/C"""
        return {k: v / 100 if k == "gamma_pdc" else v for k, v in self.dict().items()}


class CECModuleParameters(PVLibBase):
    """Parameters for the modules that make up an array in a SAM-like model"""

    alpha_sc: float = Field(
        ...,
        description=(
            "Short-circuit current temperature coefficient of the module "
            "in units of A/C"
        ),
    )
    a_ref: float = Field(
        ...,
        description=(
            "Product of number of cells in series, diode ideality factor, "
            "and thermal voltage at reference conditions"
        ),
        ge=0,
    )
    I_L_ref: float = Field(
        ...,
        description=(
            "Light-generated current (or photocurrent) at reference "
            "conditions, in amperes"
        ),
    )
    I_o_ref: float = Field(
        ...,
        description=(
            "Dark or diode reverse saturation current at reference conditions,"
            "in amperes"
        ),
    )
    R_sh_ref: float = Field(
        ...,
        description="Shunt resistance at reference conditions, in ohms",
    )
    R_s: float = Field(
        ..., description="Series resistance at reference conditions, in ohms"
    )
    gamma_r: float = Field(
        ...,
        description=(
            "Temperature coefficient of power in units of %/C. "
            "Typically -0.2 to -0.5 % per degree C"
        ),
    )
    cells_in_series: int = Field(
        ..., description="Number of cells connected in series in a module", ge=0
    )
    Adjust: float = Field(
        0.0,
        description=(
            "Factor used to adjust temperature coefficients for voltage "
            "and current to match temperature coefficient for power, percent"
        ),
    )
    EgRef: float = Field(
        1.121,
        description=(
            "Energy bandgap at reference temperature in units of eV. "
            "1.121 eV for all modules in the CEC database."
        ),
        gt=0,
    )
    dEgdT: float = Field(
        -0.0002677,
        description=(
            "The temperature dependence of the energy bandgap at reference "
            "conditions in units of 1/K. -0.0002677 1/K for all modules in "
            "the CEC database."
        ),
    )
    _modelchain_dc_model: str = PrivateAttr("cec")
    _gamma: float = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._gamma = self.gamma_r / 100

    @root_validator(skip_on_failure=True)
    def validate_diode_params(cls, values):

        try:
            pvlib.pvsystem.calcparams_cec(
                effective_irradiance=1000,
                temp_cell=25,
                **{
                    k: v
                    for k, v in values.items()
                    if k not in ("gamma_r", "cells_in_series")
                },
            )
        except Exception:  # pragma: no cover
            raise ValueError(
                "Unable to calculate single diode parameters from parameters supplied."
            )
        return values

    def pvlib_dict(self):
        """Convert to a dict pvlib understands for `module_parameters` by removing
        gamma_r"""
        return {k: v for k, v in self.dict().items() if k != "gamma_r"}


class PVsystTemperatureParameters(SPIBase):
    """Parameters for the cell temperature model of the modules in a
    PVSyst-like model"""

    u_c: float = Field(
        29.0, description="Combined heat loss factor coefficient, units of W/m^2/C"
    )
    u_v: float = Field(
        0.0,
        description=(
            "Combined heat loss factor influenced by wind, units of (W/m^2)/(C m/s)"
        ),
    )
    eta_m: float = Field(0.1, description="Module external efficiency as a fraction")
    alpha_absorption: float = Field(0.9, description="Absorption coefficient")
    _modelchain_temperature_model: str = PrivateAttr("pvsyst")


class SAPMTemperatureParameters(SPIBase):
    """Parameters for the cell temperature model of the modules in the
    Sandia Array Performance Model"""

    a: float = Field(
        ..., description="Parameter a of the Sandia Array Performance Model"
    )
    b: float = Field(
        ..., description="Parameter b of the Sandia Array Performance Model"
    )
    deltaT: float = Field(
        ..., description="Parameter delta T of the Sandia Array Performance Model"
    )
    _modelchain_temperature_model: str = PrivateAttr("sapm")


class NOCTSAMTemperatureParameters(SPIBase):
    """Parameters for the NOCT SAM temperature model"""

    noct: float = Field(
        ...,
        description=(
            "Nominal operating cell temperature [C], determined at conditions of "
            "800 W/m^2 irradiance, 20 C ambient air temperature and 1 m/s wind."
        ),
    )
    eta_m_ref: float = Field(
        ...,
        description=(
            "Module external efficiency [unitless] at reference conditions of "
            "1000 W/m^2 and 20C."
        ),
    )
    transmittance_absorptance: float = Field(
        0.9,
        description=(
            "Coefficient for combined transmittance and absorptance effects. [unitless]"
        ),
    )
    array_height: int = Field(
        1,
        description=(
            "Height of array above ground in stories (one story is about 3m). Must "
            "be either 1 or 2. For systems elevated less than one story, use 1. "
            "If system is elevated more than two stories, use 2."
        ),
        ge=1,
        le=2,
    )
    mount_standoff: float = Field(
        4,
        description=(
            "Distance between array mounting and mounting surface. Use default "
            "if system is ground-mounted. [inches]"
        ),
    )
    _modelchain_temperature_model: str = PrivateAttr("noct_sam")


class PVArray(SPIBase):
    """Parameters of a PV array that feeds into one inverter"""

    name: str = UserString("", description="Name of this array")
    make_model: str = UserString(
        "",
        title="Module Make & Model",
        description="Make and model of the PV modules in this array",
    )
    module_parameters: Union[
        PVsystModuleParameters, PVWattsModuleParameters, CECModuleParameters
    ] = Field(
        ...,
        title="Module Parameters",
        description="Parameters describing PV modules in this array",
    )
    temperature_model_parameters: Union[
        PVsystTemperatureParameters,
        SAPMTemperatureParameters,
        NOCTSAMTemperatureParameters,
    ] = Field(
        ...,
        title="Temperature Model Parameters",
        description=(
            "Parameters describing the temperature characteristics of the modules"
        ),
    )
    tracking: Union[FixedTracking, SingleAxisTracking] = Field(
        ..., description="Parameters describing single-axis tracking or fixed mounting"
    )
    albedo: float = Field(
        ..., description="Albedo of the surface around the array", ge=0
    )
    modules_per_string: int = Field(
        ...,
        title="Modules Per String",
        description="Number of PV modules per string",
        gt=0,
    )
    strings: int = Field(
        ..., description="Number of parallel strings in the array", gt=0
    )
    _modelchain_models: Tuple[Tuple[str, str], ...] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._modelchain_models = (
            ("dc_model", self.module_parameters._modelchain_dc_model),
            (
                "temperature_model",
                self.temperature_model_parameters._modelchain_temperature_model,
            ),
            ("dc_ohmic_model", "no_loss"),
        )


class PVWattsLosses(SPIBase):
    """Parameters describing the PVWatts system loss model"""

    soiling: float = Field(2.0, description="Soiling loss, %")
    shading: float = Field(3.0, description="Shading loss, %")
    snow: float = Field(0.0, description="Snow loss, %")
    mismatch: float = Field(2.0, description="Mismatch loss, %")
    wiring: float = Field(2.0, description="Wiring loss, %")
    connections: float = Field(0.5, description="Connections loss, %")
    lid: float = Field(1.5, title="LID", description="Light induced degradation, %")
    nameplate_rating: float = Field(1.0, description="Nameplate Rating loss, %")
    age: float = Field(0.0, description="Age loss, %")
    availability: float = Field(3.0, description="Availability loss, %")
    _modelchain_losses_model: str = PrivateAttr("pvwatts")


class PVWattsInverterParameters(SPIBase):
    """DC-AC power conversion parameters of an inverter for the PVWatts model"""

    pdc0: float = Field(
        ...,
        description=(
            "DC power input which produces the rated AC output power at the "
            "nominal DC voltage of the inverter, W"
        ),
    )
    eta_inv_nom: float = Field(
        0.96, description="Nominal inverter efficiency, unitless"
    )
    eta_inv_ref: float = Field(
        0.9637, description="Reference inverter efficiency, unitless"
    )
    _modelchain_ac_model: str = PrivateAttr("pvwatts")
    _pac0: float = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._pac0 = self.pdc0 * self.eta_inv_nom


class SandiaInverterParameters(SPIBase):
    """DC-AC power conversion parameters of an inverter for Sandia's
    Grid-Connected PV Inverter model"""

    Paco: float = Field(..., description="AC power rating of the inverter, W")
    Pdco: float = Field(
        ...,
        description=(
            "DC power which produces the rated AC output power at the "
            "nominal DC voltage of the inverter, W"
        ),
    )
    Vdco: float = Field(
        ...,
        description=(
            "Nominal DC voltage at which the AC power rating is determined, V"
        ),
    )
    Pso: float = Field(
        ...,
        description=(
            "DC power required to start the inversion process, assumed equal "
            "to self consumption by the inverter, W"
        ),
    )
    C0: float = Field(
        ...,
        description=(
            "Parameter defining the curvature of the relationship between AC "
            "power and DC power at reference operating conditions, 1/W"
        ),
    )
    C1: float = Field(
        ...,
        description=(
            "Empirical coefficient allowing Pdco to vary linearly with DC "
            "voltage input, 1/V"
        ),
    )
    C2: float = Field(
        ...,
        description=(
            "Empirical coefficient allowing Pso to vary linearly with DC "
            "voltage input, 1/V"
        ),
    )
    C3: float = Field(
        ...,
        description=(
            "Empirical coefficient allowing C0 to vary linearly with DC "
            "voltage input, 1/V"
        ),
    )
    Pnt: float = Field(
        ...,
        description=(
            "AC power consumed by the inverter when no AC power is exported "
            " (i.e., night tare), W"
        ),
    )
    _modelchain_ac_model: str = PrivateAttr("sandia")
    _pac0: float = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._pac0 = self.Paco


class AOIModelEnum(str, Enum):
    """Model to calculate the incidence angle modifier"""

    no_loss = "no_loss"
    physical = "physical"
    ashrae = "ashrae"
    sapm = "sapm"
    martin_ruiz = "martin_ruiz"


class SpectralModelEnum(str, Enum):
    """Spectral losses model"""

    no_loss = "no_loss"


class ClearskyModelEnum(str, Enum):
    """Model to estimate clear sky GHI, DNI, DHI"""

    ineichen = "ineichen"
    haurwitz = "haurwitz"
    simplified_solis = "simplified_solis"


class AirmassModelEnum(str, Enum):
    """Model to estimate relative airmass at sea level"""

    simple = "simple"
    kasten1966 = "kasten1966"
    youngirvine1967 = "youngirvine1967"
    kastenyoung1989 = "kastenyoung1989"
    gueymard1993 = "gueymard1993"
    young1994 = "young1994"
    pickering2002 = "pickering2002"


class TranspositionModelEnum(str, Enum):
    """Transposition model to determine total in-plane irradiance and the
    beam, sky diffuse, and ground reflected components"""

    isotropic = "isotropic"
    klucher = "klucher"
    haydavies = "haydavies"
    reindl = "reindl"
    king = "king"
    perez = "perez"


class Inverter(SPIBase):
    """Parameters for a single inverter feeding into a PV system"""

    name: str = UserString("", description="Name of this inverter")
    make_model: str = UserString(
        "",
        title="Inverter Make & Model",
        description="Make and model of the inverter",
    )
    arrays: List[PVArray] = Field(
        ...,
        description="List of PV arrays that are connected to this inverter",
        min_items=1,
    )
    losses: Optional[PVWattsLosses] = Field(
        {}, description="Parameters describing the array losses"
    )
    inverter_parameters: Union[
        PVWattsInverterParameters, SandiaInverterParameters
    ] = Field(
        ...,
        title="Inverter Parameters",
        description="Power conversion parameters for the inverter",
    )
    airmass_model: AirmassModelEnum = AirmassModelEnum.kastenyoung1989
    aoi_model: AOIModelEnum = AOIModelEnum.physical
    clearsky_model: ClearskyModelEnum = ClearskyModelEnum.ineichen
    spectral_model: SpectralModelEnum = SpectralModelEnum.no_loss
    transposition_model: TranspositionModelEnum = TranspositionModelEnum.haydavies
    _modelchain_models: Tuple[Tuple[str, str], ...] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._modelchain_models = self.arrays[0]._modelchain_models + (
            ("ac_model", self.inverter_parameters._modelchain_ac_model),
            (
                "losses_model",
                getattr(self.losses, "_modelchain_losses_model", "no_loss"),
            ),
            ("airmass_model", self.airmass_model),
            ("aoi_model", self.aoi_model),
            ("clearsky_model", self.clearsky_model),
            ("spectral_model", self.spectral_model),
            ("transposition_model", self.transposition_model),
        )

    @root_validator
    def check_only_one_array_for_tracker(cls, values):
        arrays = values.get("arrays")
        if arrays is not None and len(arrays) > 1:
            for arr in arrays:
                if isinstance(arr.tracking, SingleAxisTracking):
                    raise ValueError(
                        "Multiple arrays per inverter with any single axis "
                        "trackers is not supported"
                    )
        return values


class PVSystem(SPIBase):
    """Parameters for an entire PV system at some location"""

    name: str = UserString(
        ...,
        title="PV System Name",
        description="Name of the system",
    )
    latitude: float = Field(
        ...,
        description=(
            "Latitude of the system in degrees NORTH of the equator. Use a NEGATIVE sign "
            "for all systems in the southern hemisphere."
        ),
        ge=-90,
        le=90,
    )
    longitude: float = Field(
        ...,
        description="Longitude of the system in degrees EAST of the prime meridian (zero degrees).",
        ge=-180,
        le=180,
    )
    elevation: float = Field(
        ...,
        description=(
            "Elevation of the system in METERS above sea level. Use a NEGATIVE sign if the system "
            "is below sea level."
        ),
        ge=-300,
    )
    inverters: List[Inverter] = Field(
        ..., description="List of inverters that make up this system", min_items=1
    )

    class Config:
        schema_extra = {"example": SYSTEM_EXAMPLE}


class StoredObjectID(SPIBase):
    object_id: UUID = Field(..., description="Unique identifier of the object")
    object_type: str = Field("system", description="Type of the object")

    class Config:
        # allow extra fields to go into Stored objects as they are
        # removed when serializing. Eases putting DB objects into models
        extra = "ignore"
        schema_extra = {
            "example": {
                "object_id": "6b61d9ac-2e89-11eb-be2a-4dc7a6bcd0d9",
                "object_type": "system",
            }
        }


class StoredObject(StoredObjectID):
    created_at: dt.datetime = Field(..., description="Datetime the object was created")
    modified_at: dt.datetime = Field(
        ..., description="Datetime the object was last modified"
    )

    class Config:
        schema_extra = {
            "example": {
                "object_id": "6b61d9ac-2e89-11eb-be2a-4dc7a6bcd0d9",
                "object_type": "system",
                "created_at": "2020-12-01T01:23:00+00:00",
                "modified_at": "2020-12-01T01:23:00+00:00",
            }
        }


class StoredPVSystem(StoredObject):

    definition: PVSystem

    class Config:
        schema_extra = {
            "example": {
                "object_id": "6b61d9ac-2e89-11eb-be2a-4dc7a6bcd0d9",
                "object_type": "system",
                "created_at": "2020-12-01T01:23:00+00:00",
                "modified_at": "2020-12-01T01:23:00+00:00",
                "definition": SYSTEM_EXAMPLE,
            }
        }


class UserInfo(StoredObject):
    """Information about the current user"""

    auth0_id: str = Field(..., description="User ID from Auth 0")


class JobTimeindex(SPIBase):
    """Parameters for a time index that all data uploads must conform to.
    Data is assumed to time-averaged and closed and labeled at the left endpoint, i.e.
    a datapoint at 23:00 of data with a 1 hour time step is assumed be the
    average of data from 23:00 to 23:59.
    """

    start: dt.datetime = Field(
        ...,
        description=(
            "Start of the time range that data will be uploaded for. "
            "String values in the format YYYY-MM-DD[T]HH:MM:SS[Z or +-HH[:]MM] "
            "may be provided. "
            "Integers/floats may be provided and are assumed to be Unix time."
        ),
    )
    end: dt.datetime = Field(
        ...,
        description="End (exclusive) of the time range that data will be uploaded for",
    )
    step: dt.timedelta = Field(
        ...,
        description=(
            "Time step between each data point in whole minutes, "
            "from 1 to 60 minutes. Acceptable formats include ISO 8601 timedeltas,"
            " strings formatted like HH:MM, and integers/float assumed as seconds."
        ),
    )
    timezone: Optional[str] = Field(
        ...,
        description="Timezone data will be converted to before computation. "
        "Unlocalized data will be localized to this timezone. If timezone is "
        "null, the timezone will be inferred from start/end.",
    )
    _time_range: List[dt.datetime] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        tr = pd.date_range(start=self.start, end=self.end, freq=self.step)
        if tr[-1] == self.end:
            tr = tr[:-1]
        if tr.tzinfo is None:
            self._time_range = tr.tz_localize(
                self.timezone, ambiguous=True, nonexistent="NaT"
            )
            self._time_range = self._time_range[
                ~(self._time_range.duplicated() | self._time_range.isna())
            ]
        else:
            if self.timezone is not None:
                self._time_range = tr.tz_convert(self.timezone)
            else:
                self.timezone = str(tr.tzinfo)
                self._time_range = tr

    @root_validator(pre=True)
    def restrict_timedelta_number(cls, values):
        # restrict size of step if int/float and avoid overflowerror
        step = values.get("step")
        if isinstance(step, (int, float)):
            if abs(step) > 1e8:
                raise ValueError("Step much too large")
        return values

    @root_validator(skip_on_failure=True)
    def check_start_end_tz(cls, values):
        start = values.get("start")
        end = values.get("end")
        tz = values.get("timezone")
        if start is not None and end is not None and start > end:
            raise ValueError("'start' is after 'end'")
        if start.tzinfo != end.tzinfo:
            raise ValueError("'start' and 'end' must have the same timezone")
        if tz is None and start.tzinfo is None:
            raise ValueError("Could not infer timezone")
        return values

    @validator("step")
    def check_step(cls, v):
        secs = v.total_seconds()
        if secs < 60:
            raise ValueError("The minimum time step is 1 minute")
        elif secs > 3600:
            raise ValueError("The maximum time step is 60 minutes")
        if secs % 60 != 0:
            raise ValueError("The time step must be in whole minutes")
        return v

    @validator("timezone")
    def check_tz(cls, v):
        if v is not None and v not in TIMEZONES:
            raise ValueError("Unrecognized timezone")
        return v


class WeatherGranularityEnum(str, Enum):
    """Level of granularity of uploaded weather data"""

    system = "system"
    inverter = "inverter"
    array = "array"


class PerformanceGranularityEnum(str, Enum):
    """Level of granularity of uploaded performance data"""

    system = "system"
    inverter = "inverter"


class IrradianceTypeEnum(str, Enum):
    """Type of irradiance included in weather files"""

    standard = "standard"
    poa = "poa"
    effective = "effective"


class TemperatureTypeEnum(str, Enum):
    """Type of temperature included in weather files"""

    air = "air"
    module = "module"
    cell = "cell"


class JobDataTypeEnum(str, Enum):
    reference_weather = "reference weather data"
    actual_weather = "actual weather data"
    reference_performance = "reference performance data"
    reference_performance_dc = "reference DC performance data"
    modeled_performance = "modeled performance data"
    actual_performance = "actual performance data"
    monthly_actual_weather = "actual monthly weather data"
    monthly_reference_weather = "reference monthly weather data"
    monthly_actual_performance = "actual monthly performance data"
    monthly_reference_performance = "reference monthly performance data"


class JobDataItem(SPIBase):
    schema_path: str = Field(
        ..., description="Relative to PV system definition, i.e. /inverters/0/arrays/0"
    )
    type: JobDataTypeEnum
    _data_cols: List[str] = PrivateAttr()

    @classmethod
    def from_types(
        cls,
        schema_path: str,
        type_: JobDataTypeEnum,
        irradiance_type: Optional[IrradianceTypeEnum] = None,
        temperature_type: Optional[TemperatureTypeEnum] = None,
        **kwargs,
    ):
        """Intialization that also sets _data_cols for ease of use later when adding
        data_columns to StoredJobDataMetadata"""
        cols = [
            "time",
        ]
        if type_ in (JobDataTypeEnum.reference_weather, JobDataTypeEnum.actual_weather):
            if irradiance_type == IrradianceTypeEnum.effective:
                cols += ["effective_irradiance"]
            elif irradiance_type == IrradianceTypeEnum.poa:
                cols += ["poa_global", "poa_direct", "poa_diffuse"]
            else:
                cols += ["ghi", "dni", "dhi"]
            if temperature_type == TemperatureTypeEnum.cell:
                cols += ["cell_temperature"]
            elif temperature_type == TemperatureTypeEnum.module:
                cols += ["module_temperature"]
            else:
                cols += ["temp_air", "wind_speed"]
        elif type_ in (
            JobDataTypeEnum.reference_performance,
            JobDataTypeEnum.actual_performance,
            JobDataTypeEnum.modeled_performance,
            JobDataTypeEnum.reference_performance_dc,
        ):
            cols += ["performance"]
        elif type_ in (
            JobDataTypeEnum.monthly_actual_weather,
            JobDataTypeEnum.monthly_reference_weather,
        ):
            cols = [
                "month",
                "total_poa_insolation",
                "average_daytime_cell_temperature",
            ]
        elif type_ in (
            JobDataTypeEnum.monthly_actual_performance,
            JobDataTypeEnum.monthly_reference_performance,
        ):
            cols = ["month", "total_energy"]
        out = cls(schema_path=schema_path, type=type_, **kwargs)
        out._data_cols = cols
        return out


class JobParametersBase(SPIBase):
    system_id: UUID
    time_parameters: JobTimeindex


class CalculateMixin(SPIBase):
    # in principle, these both could be on a per model chain/inverter basis,
    # but easier to just keep everything the same for the system
    irradiance_type: IrradianceTypeEnum
    temperature_type: TemperatureTypeEnum
    weather_granularity: WeatherGranularityEnum
    _weather_types: Tuple[JobDataTypeEnum, ...] = PrivateAttr()

    def _construct_data_items(
        self, system: PVSystem
    ) -> Dict[Tuple[str, JobDataTypeEnum], JobDataItem]:
        if self.weather_granularity == WeatherGranularityEnum.system:
            weather_paths = ["/"]
        elif self.weather_granularity == WeatherGranularityEnum.inverter:
            weather_paths = [f"/inverters/{i}" for i in range(len(system.inverters))]
        elif self.weather_granularity == WeatherGranularityEnum.array:
            weather_paths = [
                f"/inverters/{i}/arrays/{j}"
                for i, inv in enumerate(system.inverters)
                for j in range(len(inv.arrays))
            ]

        out = {
            (wp, jt): JobDataItem.from_types(
                schema_path=wp,
                type_=jt,
                irradiance_type=getattr(self, "irradiance_type", None),
                temperature_type=getattr(self, "temperature_type", None),
            )
            for jt in self._weather_types
            for wp in weather_paths
        }
        return out


class CompareMixin(CalculateMixin):
    performance_granularity: PerformanceGranularityEnum
    _performance_types: Tuple[JobDataTypeEnum, ...] = PrivateAttr()

    def _construct_data_items(
        self, system: PVSystem
    ) -> Dict[Tuple[str, JobDataTypeEnum], JobDataItem]:
        out = super()._construct_data_items(system)
        if self.performance_granularity == PerformanceGranularityEnum.system:
            perf_paths = ["/"]
        elif self.performance_granularity == PerformanceGranularityEnum.inverter:
            perf_paths = [f"/inverters/{i}" for i in range(len(system.inverters))]
        else:  # can be None
            perf_paths = []
        out.update(
            {
                (pp, jt): JobDataItem.from_types(
                    schema_path=pp,
                    type_=jt,
                    irradiance_type=getattr(self, "irradiance_type", None),
                    temperature_type=getattr(self, "temperature_type", None),
                )
                for jt in self._performance_types
                for pp in perf_paths
            }
        )
        return out


class CalculateEnum(str, Enum):
    reference_performance = "reference performance"
    modeled_performance = "modeled performance"


class CalculatePerformanceJobParameters(CalculateMixin, JobParametersBase):
    """Calculate the given type of performance"""

    calculate: CalculateEnum

    def __init__(self, **data):
        super().__init__(**data)
        if self.calculate == CalculateEnum.reference_performance:
            self._weather_types = (JobDataTypeEnum.reference_weather,)
        else:
            self._weather_types = (JobDataTypeEnum.actual_weather,)


class ModeledActualEnum(str, Enum):
    modeled_actual = "modeled and actual performance"


class CompareModeledActualJobParameters(CompareMixin, JobParametersBase):
    """Calculate and compare modeled to actual performance"""

    compare: ModeledActualEnum
    _weather_types: Tuple[JobDataTypeEnum, ...] = PrivateAttr(
        (JobDataTypeEnum.actual_weather,)
    )
    _performance_types: Tuple[JobDataTypeEnum, ...] = PrivateAttr(
        (JobDataTypeEnum.actual_performance,)
    )


class WeatherPREnum(str, Enum):
    weather_adjusted_pr = "weather-adjusted performance ratio"


class CalculateWeatherAdjustedPRJobParameters(CompareMixin, JobParametersBase):
    """Calculate the weather-adjusted performance ratio"""

    calculate: WeatherPREnum
    _weather_types = PrivateAttr((JobDataTypeEnum.actual_weather,))
    _performance_types = PrivateAttr((JobDataTypeEnum.actual_performance,))


MODEL_CHAIN_METHOD_MAP = {
    IrradianceTypeEnum.effective: "run_model_from_effective_irradiance",
    IrradianceTypeEnum.poa: "run_model_from_poa",
    IrradianceTypeEnum.standard: "run_model",
}


class ActualDataParams(CompareMixin):
    """Parameters for the "actual" data series"""

    _weather_types = PrivateAttr((JobDataTypeEnum.actual_weather,))
    _performance_types = PrivateAttr((JobDataTypeEnum.actual_performance,))
    _model_chain_method: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._model_chain_method = MODEL_CHAIN_METHOD_MAP[self.irradiance_type]


class ReferenceDataEnum(str, Enum):
    weather_and_ac = "weather and AC performance"
    weather_and_ac_and_dc = "weather, AC, and DC performance"
    weather_only = "weather only"


class ReferenceDataParams(CompareMixin):
    """Parameters for the "reference" data series"""

    data_available: ReferenceDataEnum
    performance_granularity: Optional[PerformanceGranularityEnum]  # type: ignore
    _weather_types = PrivateAttr((JobDataTypeEnum.reference_weather,))
    _model_chain_method: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        if self.data_available == ReferenceDataEnum.weather_and_ac:
            self._performance_types = (JobDataTypeEnum.reference_performance,)
        elif self.data_available == ReferenceDataEnum.weather_and_ac_and_dc:
            self._performance_types = (
                JobDataTypeEnum.reference_performance,
                JobDataTypeEnum.reference_performance_dc,
            )
        else:
            self._performance_types = ()
        self._model_chain_method = MODEL_CHAIN_METHOD_MAP[self.irradiance_type]

    @root_validator
    def check_performance_granularity(cls, values):
        da = values.get("data_available")
        pg = values.get("performance_granularity")
        if da == ReferenceDataEnum.weather_only and pg is not None:
            raise ValueError(
                "Performance granularity is invalid when not providing reference "
                "performance"
            )
        return values


class ReferenceActualEnum(str, Enum):
    reference_actual = "reference and actual performance"


class CompareReferenceActualJobParameters(JobParametersBase):
    """Compare reference to actual performance"""

    reference_data_parameters: ReferenceDataParams
    actual_data_parameters: ActualDataParams
    compare: ReferenceActualEnum

    def _construct_data_items(
        self, system_definition: PVSystem
    ) -> Dict[Tuple[str, JobDataTypeEnum], JobDataItem]:
        out = self.reference_data_parameters._construct_data_items(system_definition)
        out.update(self.actual_data_parameters._construct_data_items(system_definition))
        return out


class MonthlyReferenceActualEnum(str, Enum):
    monthly_reference_actual = "monthly reference and actual performance"


class CompareMonthlyReferenceActualJobParameters(SPIBase):
    """Compare reference to actual performance on a monthly time
    scale. Data is expected to be at the system level and include
    monthly insolation, energy, and average daytime temperature.
    """

    system_id: UUID
    compare: MonthlyReferenceActualEnum

    def _construct_data_items(
        self, system_definition: PVSystem
    ) -> Dict[Tuple[str, JobDataTypeEnum], JobDataItem]:
        return {
            ("/", jt): JobDataItem.from_types(schema_path="/", type_=jt)
            for jt in (
                JobDataTypeEnum.monthly_reference_weather,
                JobDataTypeEnum.monthly_actual_weather,
                JobDataTypeEnum.monthly_reference_performance,
                JobDataTypeEnum.monthly_actual_performance,
            )
        }


class ModeledDataParams(CalculateMixin):
    """Parameters for the "modeled" data series"""

    _weather_types = PrivateAttr((JobDataTypeEnum.actual_weather,))
    _model_chain_method: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._model_chain_method = MODEL_CHAIN_METHOD_MAP[self.irradiance_type]


class ReferenceModeledEnum(str, Enum):
    reference_modeled = "reference and modeled performance"


class CompareReferenceModeledJobParameters(JobParametersBase):
    """Compare reference to modeled performance"""

    reference_data_parameters: ReferenceDataParams
    modeled_data_parameters: ModeledDataParams
    compare: ReferenceModeledEnum

    def _construct_data_items(
        self, system_definition: PVSystem
    ) -> Dict[Tuple[str, JobDataTypeEnum], JobDataItem]:
        out = self.reference_data_parameters._construct_data_items(system_definition)
        out.update(
            self.modeled_data_parameters._construct_data_items(system_definition)
        )
        return out


JobParametersType = Union[
    CompareReferenceActualJobParameters,
    CompareReferenceModeledJobParameters,
    CompareModeledActualJobParameters,
    CompareMonthlyReferenceActualJobParameters,
    CalculateWeatherAdjustedPRJobParameters,
    CalculatePerformanceJobParameters,
]

JOB_PARAMS_EXAMPLE = dict(
    system_id=SYSTEM_ID,
    compare="modeled and actual performance",
    time_parameters=dict(
        start="2020-01-01T00:00:00+00:00",
        end="2020-12-31T23:59:59+00:00",
        step="15:00",
        timezone="UTC",
    ),
    weather_granularity="array",
    irradiance_type="poa",
    temperature_type="module",
    performance_granularity="inverter",
)


JOB_DATA_META_EXAMPLE = dict(
    schema_path="/inverters/0",
    type="actual performance data",
    filename="inverter_0_performance.arrow",
    data_format="application/vnd.apache.arrow.file",
    present=True,
    data_columns=["time", "performance"],
)


class JobDataMetadata(JobDataItem):
    filename: str = Field("", description="Filename of the uploaded file")
    data_format: str = Field("", description="Format of the binary file")
    present: bool = Field(False, description="If the data has been uploaded or not")
    data_columns: List[str] = Field(
        [], description="Column names the data is expected to have"
    )


class StoredJobDataMetadata(StoredObject):
    definition: JobDataMetadata


JOB_EXAMPLE = dict(system_definition=SYSTEM_EXAMPLE, parameters=JOB_PARAMS_EXAMPLE)


class Job(SPIBase):
    # duplicated here to track without worrying about changes
    system_definition: PVSystem
    parameters: JobParametersType
    _data_items: Dict[Tuple[str, JobDataTypeEnum], JobDataItem] = PrivateAttr()
    _model_chain_method: Optional[str] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._data_items = self.parameters._construct_data_items(self.system_definition)
        # determine what columns to expect in uploads
        irradiance_type = getattr(self.parameters, "irradiance_type", None)
        self._model_chain_method = MODEL_CHAIN_METHOD_MAP.get(irradiance_type, None)


class JobStatusEnum(str, Enum):
    incomplete = "incomplete"
    prepared = "prepared"
    queued = "queued"
    running = "running"
    complete = "complete"
    error = "error"


class JobStatus(SPIBase):
    status: JobStatusEnum = Field(
        ...,
        description="""Status of the job:
- incomplete: The job has been created but missing required data.
- prepared: The job has been created and all required data is present.
- queued: The job has been queued for execution.
- running: The job is running.
- complete: The job has completed without fatal errors and results are ready.
- error: The job encountered a fatal error. The results will describe the error.
""",
    )
    last_change: dt.datetime


class StoredJob(StoredObject):
    definition: Job
    status: JobStatus
    data_objects: List[StoredJobDataMetadata]

    class Config:
        schema_extra = {
            "example": {
                "object_id": "e1772e64-43ac-11eb-92c2-f4939feddd82",
                "object_type": "job",
                "created_at": "2020-12-11T19:52:00+00:00",
                "modified_at": "2020-12-11T19:52:00+00:00",
                "definition": JOB_EXAMPLE,
                "status": {
                    "status": "incomplete",
                    "last_change": "2020-12-11T20:00:00+00:00",
                },
                "data_objects": [
                    {
                        "object_id": "ecaa5a40-43ac-11eb-a75d-f4939feddd82",
                        "object_type": "job_data",
                        "created_at": "2020-12-11T19:52:00+00:00",
                        "modified_at": "2020-12-11T19:52:00+00:00",
                        "definition": {
                            "schema_path": "/inverters/0/arrays/0",
                            "type": "reference weather data",
                            "present": False,
                            "data_columns": [
                                "time",
                                "poa_global",
                                "poa_direct",
                                "poa_diffuse",
                                "module_temperature",
                            ],
                        },
                    },
                    {
                        "object_id": "f9ef0c00-43ac-11eb-8931-f4939feddd82",
                        "object_type": "job_data",
                        "created_at": "2020-12-11T19:52:00+00:00",
                        "modified_at": "2020-12-11T20:00:00+00:00",
                        "definition": JOB_DATA_META_EXAMPLE,
                    },
                ],
            }
        }


class DataPeriods(SPIBase):
    expected: str = Field(..., description="Expected period of the data")
    uploaded: str = Field(..., description="Most common period of the uploaded data")


class DataParsingStats(SPIBase):
    number_of_expected_rows: int = Field(
        ..., description="Number of total rows expected in the data upload."
    )
    number_of_extra_rows: int = Field(
        ...,
        description="Number of rows outside the specified time parameters for the job.",
    )
    number_of_missing_rows: int = Field(
        ..., description="Number of rows that were missing but expected in the upload."
    )
    data_periods: DataPeriods
    extra_times: List[dt.datetime] = Field(
        ...,
        description=(
            "Times that were included in the upload but are outside the "
            "job time parameters."
        ),
    )
    missing_times: List[dt.datetime] = Field(
        ...,
        description=(
            "Times that were modeled based on the job time parameters "
            "but missing from the upload."
        ),
    )
    number_of_missing_values: Dict[str, int] = Field(
        ...,
        description=(
            "Number of values in each column that were missing upon upload. "
            "Does not include whole rows that were missing."
        ),
    )


class JobResultTypeEnum(str, Enum):
    performance = "performance data"
    weather = "weather data"
    error = "error message"
    monthy_summary = "monthly summary"
    daytime_flag = "daytime flag"
    actual_vs_modeled = "actual vs modeled energy"
    weather_adjusted_performance = "weather adjusted performance"
    actual_vs_adjusted_reference = "actual vs weather adjusted reference"
    modeled_vs_adjusted_reference = "modeled vs weather adjusted reference"
    # will need other types for performance ratio etc.


class JobResultMetadata(SPIBase):
    type: JobResultTypeEnum = Field(
        ...,
        description="""Type of data in this result:
- performance data: AC performance data at the level (system or inverter) given by
  schema_path. Data has columns are time and performance.
- weather data: Modeled weather/environment data for the array given in schema_path.
  Data has columns time, global plane-of-array irradiance (poa_global), and
  cell temperature.
- monthly summary: Monthly total energy (Wh), plane of array insolation (Wh/m^2),
  effective insolation (Wh/m^2), and average daytime cell temperature.
- actual vs modeled energy: Monthly totals of actual energy (Wh), modeled energy (Wh),
  the difference (actual - modeled) (Wh), and the ratio of actual / modeled.
- weather adjusted performance: AC performance adjusted for differences in weather
  conditions at the level (system or inverter) given by  schema_path. Data has
  columns are time and performance.
- actual vs adjusted reference: Monthly totals of actual energy (Wh), weather adjusted
  reference energy (Wh), the difference (actual - reference) (Wh), and the ratio of
  actual / reference.
- modeled vs adjusted reference: Monthly totals of modeled energy (Wh), weather
  adjusted  reference energy (Wh), the difference (modeled - reference) (Wh), and
  the ratio of modeled / reference.
- daytime flag: boolean, 1 if the timestamp is day-time defined as the when the
  solar zenith for the midpoint of the interval is < 87.0 degrees.
- error message: The result could not be computed. The result for this object will
  be a JSON object describing the error.
""",
    )
    schema_path: str = Field(
        ..., description="Relative to PV system definition, i.e. /inverters/0/arrays/0"
    )
    data_format: str = Field(..., description="Format of the binary data")


STORED_JOB_RESULT_EXAMPLES = [
    {
        "object_id": "d84bdf30-55f2-11eb-a03d-f4939feddd82",
        "object_type": "job_result",
        "created_at": "2021-01-12T13:05:00+00:00",
        "modified_at": "2021-01-12T13:05:00+00:00",
        "definition": {
            "schema_path": "/inverters/0/arrays/0",
            "type": "weather data",
            "data_format": "application/vnd.apache.arrow.file",
        },
    },
    {
        "object_id": "e525466a-55f2-11eb-a03d-f4939feddd82",
        "object_type": "job_result",
        "created_at": "2021-01-12T13:05:00+00:00",
        "modified_at": "2021-01-12T13:05:00+00:00",
        "definition": {
            "schema_path": "/inverters/0",
            "type": "performance data",
            "data_format": "application/vnd.apache.arrow.file",
        },
    },
    {
        "object_id": "e566a59c-55f2-11eb-a03d-f4939feddd82",
        "object_type": "job_result",
        "created_at": "2021-01-12T13:05:00+00:00",
        "modified_at": "2021-01-12T13:05:00+00:00",
        "definition": {
            "schema_path": "/",
            "type": "performance data",
            "data_format": "application/vnd.apache.arrow.file",
        },
    },
]


class StoredJobResultMetadata(StoredObject):
    definition: JobResultMetadata

    class Config:
        schema_extra = {"example": STORED_JOB_RESULT_EXAMPLES}
